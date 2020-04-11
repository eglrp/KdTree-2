
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <fmt/format.h>
#include <fmt/printf.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cho_util/core/random.hpp>

std::vector<int> anchor_cache;

template <typename Scalar = float, int N = 3>
struct KdTree {
  /**
   * L2 distance.
   */
  inline Scalar Distance(Scalar* const a, Scalar* const b) const {
    Scalar out;
    for (int i = 0; i < N; ++i) {
      out += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return out;
  }

  inline void SplitLevel(const int level,
                         std::vector<int>* const indices) const {
    const int axis = level % N;

    // For now, assume multiple of 2.
    const int num_nodes = (1 << level);
    const int size = this->size >> level;
    const int half_size = size >> 1;

    // 0 ~ 512 -> fix 256
    // then fix 128, 384

    // Build subtree.
    for (int i = 0; i < num_nodes; ++i) {
      const auto i0 = indices->begin() + (i * size);
      const auto i1 = std::min(indices->end(), i0 + size);
      const auto im = i0 + half_size;
      // fmt::print("Anchor @ {}\n", std::distance(indices->begin(), im));
      // fmt::print("{} ~ {} = {}\n", std::distance(indices->begin(), i0),
      //           std::distance(indices->begin(), i1), axis);
      // 0 ~ 255, [256] 257 ~ 512

      // if (i0 != indices->begin()) {
      //  fmt::print("skip @ {}\n", std::distance(indices->begin(), i0));
      //}

      // for (const auto& a : anchor_cache) {
      //  const auto lhs =
      //      std::distance(indices->begin(), i0 + (i0 != indices->begin()));
      //  const auto rhs = std::distance(indices->begin(), i1);
      //  if (lhs <= a && a < rhs) {
      //    throw std::logic_error("Reprocessing anchor");
      //  }
      //}

      // TODO(yycho0108): Non-hacky way to deal with skipping anchors???
      std::nth_element(i0 + (i0 != indices->begin()), im, i1,
                       [this, &axis](const int lhs, const int rhs) {
                         return data[lhs * N + axis] < data[rhs * N + axis];
                       });
      // anchor_cache.emplace_back(std::distance(indices->begin(), im));

      // Local validation
      // for (auto it = i0 + (i0 != indices->begin()); it != im; ++it) {
      //  if (data[*it * N + axis] > data[*im * N + axis]) {
      //    throw std::logic_error("Local validation");
      //  }
      //}
      // for (auto it = im; it != i1; ++it) {
      //  if (data[*it * N + axis] < data[*im * N + axis]) {
      //    throw std::logic_error("Local validation");
      //  }
      //}
    }
    // SplitLevel(level + 1, indices);
    return;
  }

  bool Validate(const int level, std::vector<int>* const indices) const {
    const int axis = level % N;

    // For now, assume multiple of 2.
    const int num_nodes = (1 << level);
    const int size = this->size >> level;
    const int half_size = size >> 1;

    // Terminate by leaf size criterion.
    if (size < 10) {
      return true;
    }

    for (int i = 0; i < num_nodes; ++i) {
      const auto i0 = indices->begin() + (i * size);
      const auto i1 = std::min(indices->end(), i0 + size);
      const auto im = i0 + half_size;

      // Validation
      for (auto it = i0 + (i0 != indices->begin()); it != im; ++it) {
        if (data[*it * N + axis] > data[*im * N + axis]) {
          fmt::print("{} > {}\n", std::distance(indices->begin(), it),
                     std::distance(indices->begin(), im));

          throw std::logic_error(fmt::format(
              "NoV {} im={}", level, std::distance(indices->begin(), im)));
        }
      }
      for (auto it = im; it != i1; ++it) {
        if (data[*it * N + axis] < data[*im * N + axis]) {
          fmt::print("{} < {}\n", std::distance(indices->begin(), it),
                     std::distance(indices->begin(), im));
          throw std::logic_error(fmt::format(
              "NoV {} im={}", level, std::distance(indices->begin(), im)));
        }
      }
    }
    return Validate(level + 1, indices);
  }

  inline void GetExtents(const int index, const int level,
                         std::array<float, 2>* const pmin,
                         std::array<float, 2>* const pmax) const {
    // pmin-pmax should be populated with global min-max already.
    int parent = size >> 1;
    for (int i = 0; i < level; ++i) {
      const int axis = i % N;

      // Determine if before or after anchor.
      const int range = (size >> (i + 1));  // 8>>1 == 4
      const bool sel = index & range;       // e.g. 7 = 111 == [>=4 | >=2 | >=1]
      // fmt::print("parent {}\n", parent);
      // fmt::print("axis {}\n", axis);
      // fmt::print("index {} range {} sel : {}\n", index, range, sel);
      (*(sel ? pmin : pmax))[axis] = data[parent * N + axis];
      parent += (sel ? -1 : 1) * (range >> 1);
    }
  }

  KdTree(Scalar* const data, int size, int leaf_size = 16)
      : data(data), size(size), leaf_size(leaf_size) {
    indices.resize(size);
    std::iota(indices.begin(), indices.end(), 0);

    const int depth = std::ceil(std::log2((size + leaf_size - 1) / leaf_size));
    fmt::print("Depth : {}\n", depth);
    for (int i = 0; i < depth; ++i) {
      SplitLevel(i, &indices);
    }
  }

  Scalar* data;
  int size;
  int leaf_size;
  std::vector<int> indices;
};

void DrawKdTreeAtLevel(const KdTree<float, 2>& tree, cv::Mat* const img,
                       const int level, int i0, int i1,
                       std::array<float, 2>& pmin, std::array<float, 2>& pmax,
                       const std::function<cv::Point2f(float, float)>& pmap) {
  // if (level >= 4) {
  //  return;
  //}
  static constexpr const int kDim = 2;
  const int num_points = tree.indices.size();
  const auto& indices = tree.indices;

  const int axis = level % kDim;
  const int num_nodes = (1 << level);
  const int size = num_points >> level;
  const int half_size = size >> 1;
  const bool is_leaf = size <= tree.leaf_size;

  if (is_leaf) {
    for (auto it = i0; it != i1; ++it) {
      const auto& x = tree.data[*(indices.begin() + it) * kDim];
      const auto& y = tree.data[*(indices.begin() + it) * kDim + 1];
      cv::circle(*img, pmap(x, y), 1, cv::Scalar::all(255));
    }
    return;
  }

  // Draw sub-tree.
  {
    const auto im = (i0 + half_size);
    const auto& x = tree.data[indices[im] * kDim];
    const auto& y = tree.data[indices[im] * kDim + 1];
    fmt::print("Anchor2 @ {} = ({},{})\n", im, x, y);
    const float median = tree.data[indices[im] * kDim + axis];

    float mn = pmin[axis];
    float mx = pmax[axis];

    pmax[axis] = median;
    DrawKdTreeAtLevel(tree, img, level + 1, i0, im, pmin, pmax, pmap);
    pmax[axis] = mx;

    pmin[axis] = median;
    DrawKdTreeAtLevel(tree, img, level + 1, im, i1, pmin, pmax, pmap);
    pmin[axis] = mn;

    // Draw the splitting line.
    {
      pmin[axis] = median;
      pmax[axis] = median;
      cv::line(*img, pmap(pmin[0], pmin[1]), pmap(pmax[0], pmax[1]),
               cv::Scalar::all(255));
      pmin[axis] = mn;
      pmax[axis] = mx;
    }

    // Draw the median point.
    {
      const auto& x = tree.data[indices[im] * kDim];
      const auto& y = tree.data[indices[im] * kDim + 1];
      cv::circle(*img, pmap(x, y), 5, cv::Scalar::all(255), 1);
    }
  }
}
void DrawKdTree2d(const KdTree<float, 2>& tree, cv::Mat* const img) {
  static constexpr const int kDim = 2;
  const int num_points = tree.indices.size();

  // Compute extents for visualization.
  std::array<float, kDim> pmin, pmax;
  std::fill(pmin.begin(), pmin.end(), std::numeric_limits<float>::max());
  std::fill(pmax.begin(), pmax.end(), std::numeric_limits<float>::lowest());
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < kDim; ++j) {
      pmin[j] = std::min(pmin[j], tree.data[i * kDim + j]);
      pmax[j] = std::max(pmax[j], tree.data[i * kDim + j]);
    }
  }

  // fmt::print("pmin {}\n", fmt::join(pmin, " "));
  // fmt::print("pmax {}\n", fmt::join(pmax, " "));
  // tree.GetExtents(1, 2, &pmin, &pmax);
  // fmt::print("pmin {}\n", fmt::join(pmin, " "));
  // fmt::print("pmax {}\n", fmt::join(pmax, " "));

  // Center
  std::array<float, kDim> pctr;
  pctr[0] = 0.5 * (pmin[0] + pmax[0]);
  pctr[1] = 0.5 * (pmin[1] + pmax[1]);

  const float scale = 1.25 * std::max(pmax[0] - pmin[0], pmax[1] - pmin[1]);
  const float iscale = 1.0 / scale;
  const float ix = img->cols * iscale;
  const float iy = img->rows * iscale;

  // pmid = img->cols/2, img->rows/2
  const float cx = pctr[0] - 0.5 * scale;
  const float cy = pctr[1] - 0.5 * scale;

  const std::function<cv::Point2f(float, float)>& pmap =
      [ix, cx, iy, cy](float x, float y) -> cv::Point2f {
    return {(x - cx) * ix, (y - cy) * iy};
  };

  for (int i = 0; i < num_points; ++i) {
    const auto& x = tree.data[i * kDim];
    const auto& y = tree.data[i * kDim + 1];
    cv::circle(*img, pmap(x, y), 1, cv::Scalar::all(255));
  }

  DrawKdTreeAtLevel(tree, img, 0, 0, tree.indices.size(), pmin, pmax, pmap);
  return;
}

int main() {
  constexpr const int kNumPoints(256);
  constexpr const int kDim = 2;

  std::vector<float> dat(kNumPoints * kDim);

  // Fill with random data
  auto& rng = cho_util::core::RNG::GetInstance();
  std::generate(dat.begin(), dat.end(), [&rng]() { return rng.Randu(); });

  // std::iota(dat.begin(), dat.end(), 0);
  KdTree<float, kDim> tree(dat.data(), kNumPoints, 8);
  cv::Mat img(512, 512, CV_8UC1);
  DrawKdTree2d(tree, &img);
  cv::imshow("tree", img);
  cv::waitKey(0);
  // for (const auto& i : tree.indices) {
  //  fmt::print("{}\n", i);
  //}

  // Draw ...
}
