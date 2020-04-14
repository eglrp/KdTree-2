#include <algorithm>
#include <deque>
#include <functional>
#include <numeric>
#include <queue>
#include <vector>

#include <bits/stdc++.h>

#include <fmt/format.h>
#include <fmt/printf.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "nanoflann.hpp"

#include <cho_util/core/random.hpp>
#include <cho_util/util/timer.hpp>
#include <cho_util/vis/embedded_viewer.hpp>
#include <cho_util/vis/render_data.hpp>

#include <valgrind/callgrind.h>

#include "heap.hpp"

std::vector<int> anchor_cache;

template <typename A, typename B>
std::ostream& operator<<(std::ostream& os, const std::pair<A, B>& p) {
  return os << p.first << ',' << p.second;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::priority_queue<T>& q) {
  std::priority_queue<T> cp(q);
  while (!cp.empty()) {
    os << cp.top() << "|";
    cp.pop();
  }
  return os;
}

template <typename Scalar = float, int Dim = 1>
struct PointerAdaptor {
 private:
  const Scalar* const data;
  int size;

 public:
  using coord_t = Scalar;

  /// The constructor that sets the data set source
  PointerAdaptor(const Scalar* const data, int size) : data(data), size(size) {}

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return size; }

  inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
    return data[idx * Dim + dim];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }
};

struct Topk {
  Topk(int size) : size_(size) { values.reserve(size); }
  std::vector<std::pair<float, int>> values;
  int size_;

  inline void Reorder() {
    std::sort(values.begin(), values.end());
    // auto it = std::max_element(values.begin(), values.end());
    // std::swap(*it, values.back());
  }

  inline void Add(const std::pair<float, int>& value) {
    if (values.size() < size_) {
      values.emplace_back(value);
      if (values.size() == size_) {
        Reorder();
      }
    } else {
      if (value.first >= values.back().first) {
        return;
      }
      auto it = std::lower_bound(values.begin(), values.end(), value);
      values.back() = value;
      std::sort(it, values.end());
      // Reorder();
    }
  }

  inline int size() const { return values.size(); }

  inline const std::pair<float, int>& top() const { return values.back(); }
};

struct TopKV2 {
  TopKV2(int size) : size_(size) { values.reserve(size); }
  std::vector<std::pair<float, int>> values;
  int size_;

  inline void Add(std::pair<float, int>&& value) {
    if (values.size() < size_) {
#if 1
      values.emplace_back(std::move(value));
      if (values.size() == size_) {
        std::make_heap(values.data(), values.data() + size_);
      }
#else
      values.emplace_back(std::move(value));
      std::push_heap(values.begin(), values.end());
#endif
    } else {
      if (value >= values[0]) {
        return;
      }
      values[0] = std::move(value);
      heapreplace(values.data(), values.data() + size_);
    }
  }

  // inline void Add(const std::pair<float, int>& value) {
  //  if (values.size() < size_) {
  //    values.emplace_back(value);
  //    if (values.size() == size_) {
  //      std::make_heap(values.begin(), values.end());
  //    }
  //  } else {
  //    if (value >= values[0]) {
  //      return;
  //    }
  //    values[0] = value;
  //    heapreplace(values.begin(), values.end());
  //  }
  //}

  inline const auto size() const { return values.size(); }
  inline const std::pair<float, int>& top() const { return values[0]; }
};

struct TopKV3 {
  TopKV3(int size) : size_(size) { values.reserve(size + 1); }
  std::vector<std::pair<float, int>> values;
  int size_;

  inline void Add(std::pair<float, int>&& value) {
    if (values.size() < size_) {
      values.emplace_back(value);
      if (values.size() == size_) {
        std::sort(values.begin(), values.end());
      }
    } else {
      if (value >= values.back()) {
        return;
      }

      // for (auto it = values.rend(); it != values.rbegin(); ++it) {
      //  if (value < *it) {
      //    std::swap(value, *it);
      //  }
      //}
      // auto it = std::lower_bound(values.begin(), values.end(), value);

#if 1
      auto it = std::lower_bound(values.rbegin(), values.rend(), value,
                                 std::greater<std::pair<float, int>>());
      values.insert(it.base(), value);
      values.pop_back();
#else
      for (auto& v : values) {
        if (v > value) {
          std::swap(v, value);
        }
      }
#endif

      // values.back()=value;
      // std::sort(it,values.end());
    }
  }

  inline int size() const { return values.size(); }
  inline const std::pair<float, int>& top() const { return values.back(); }
};

template <typename Scalar = float, int N = 3>
struct KdTree {
  /**
   * L2 distance.
   */
  inline constexpr Scalar Distance(const Scalar* const a,
                                   const Scalar* const b) const {
    Scalar out{0};
    for (int i = 0; i < N; ++i) {
#if 0
      const Scalar d = a[i] - b[i];
      out += d * d;
#else
      out += (a[i] - b[i]) * (a[i] - b[i]);
#endif
    }
    return out;
  }

  inline void SplitLevel(const int level,
                         std::vector<int>* const indices) const {
    // For now, assume multiple of 2.
    const int axis = level % N;
    const int num_nodes = (1 << level);
    const int size = this->size >> level;
    const int half_size = size >> 1;
    const Scalar* const dax = data + axis;

    // Build subtree.
    for (int i = 0; i < num_nodes; ++i) {
      const auto i0 = indices->begin() + (i * size);
      const auto i1 = std::min(indices->end(), i0 + size);
      const auto im = i0 + half_size;

      // To ensure fair split, lesser/greater will depend on <parent>
      // such that nodes on the same side of the index as the parent will lie on
      // the same side of the hyperplane as the parent.
      std::nth_element(i0 + 1, im, i1,
                       [dax, &axis](const int lhs, const int rhs) {
                         return dax[lhs * N] < dax[rhs * N];
                       });
      const bool lt = dax[*i0 * N] < dax[*im * N];
      if (!lt) {
        // median will be the same, so just swap lhs and rhs.
        // now lhs is greater, rhs is less.
        std::swap_ranges(i0 + 1, im, im + 1);
      }
    }
  }

  inline bool SearchLeaf(const Scalar* const point, const int imin,
                         const int imax, TopKV2* const q) const {
    // fmt::print("{}~{}\n", imin, imax);
    // Search leaf.
    for (int i = imin; i < imax; ++i) {
      const int index = indices[i];
      const Scalar d = Distance(&data[index * N], point);
      // Insert element to priority queue.
      q->Add({d, index});
    }
  }

  void VisitBoxesAtLevel(
      int anchor, int level, std::array<Scalar, N>& pmin,
      std::array<Scalar, N>& pmax,
      const std::function<void(const std::array<Scalar, N>&,
                               std::array<Scalar, N>&)>& on_box) const {
    const int step = size >> (level + 2);
    if (0) {
      on_box(pmin, pmax);
    } else {
      // Actual bounds.
      std::array<Scalar, N> bmin = pmax, bmax = pmin;
      std::fill(bmin.begin(), bmin.end(), std::numeric_limits<float>::max());
      std::fill(bmax.begin(), bmax.end(), std::numeric_limits<float>::lowest());
      for (int i = anchor - 2 * step; i < anchor + 2 * step; ++i) {
        for (int j = 0; j < N; ++j) {
          bmin[j] = std::min(bmin[j], data[indices[i] * N + j]);
          bmax[j] = std::max(bmax[j], data[indices[i] * N + j]);
        }
      }
      on_box(bmin, bmax);
    }

    // Skip leaf node.
    if (level >= depth) {
      return;
    }

    // >> Non-leaf.
    const int axis = level % N;
    const Scalar* const dax = data + axis;
    float med = dax[indices[anchor] * N];

    // Determine search order.
    // NOTE(yycho0108): Repeating d2p compute.
    const bool anchor_sign = dax[indices[anchor + 1] * N] >= med;

    Scalar& lim_lhs = (anchor_sign ? pmax : pmin)[axis];
    Scalar& lim_rhs = (anchor_sign ? pmin : pmax)[axis];

    // Consider reviving this
    const bool search_rhs = true;

    Scalar& lim_top = search_rhs ? lim_rhs : lim_lhs;
    Scalar& lim_bot = search_rhs ? lim_lhs : lim_rhs;

    // Update bounds and search first child.
    std::swap(lim_top, med);
    VisitBoxesAtLevel(anchor + (search_rhs ? step : -step), level + 1, pmin,
                      pmax, on_box);
    std::swap(lim_top, med);

    // Update bounds and search second child.
    std::swap(lim_bot, med);
    VisitBoxesAtLevel(anchor + (search_rhs ? -step : step), level + 1, pmin,
                      pmax, on_box);
    std::swap(lim_bot, med);
  }

  void VisitBoxes(
      const std::function<void(const std::array<Scalar, N>&,
                               std::array<Scalar, N>&)>& on_box) const {
    VisitBoxesAtLevel(size >> 1, 0, const_cast<std::array<Scalar, N>&>(pmin),
                      const_cast<std::array<Scalar, N>&>(pmax), on_box);
  }

  void SearchNearestNeighborRecursiveLevel(const Scalar* const point, int k,
                                           int anchor, int level,
                                           std::array<Scalar, N>& pmin,
                                           std::array<Scalar, N>& pmax,
                                           TopKV2* const q) const {
    // Determine distance to bounding box.
    Scalar d{0};
    for (int i = 0; i < N; ++i) {
      const Scalar delta =
          std::max({pmin[i] - point[i], Scalar{0.0f}, point[i] - pmax[i]});
      d += delta * delta;
    }

    // If too far, then skip.
    if (q->size() >= k && d >= q->top().first) {
      return;
    }

    const int step = size >> (level + 2);
    // Process leaf node.
    if (level >= depth) {
      // fmt::print(":) {}\n", anchor & leaf_size);
      SearchLeaf(point, anchor - 2 * step, anchor + 2 * step, q);
      return;
    }

    // >> Non-leaf.
    const int axis = level % N;
    const Scalar* const dax = data + axis;
    float med = dax[indices[anchor] * N];

    // Determine search order.
    // NOTE(yycho0108): Repeating d2p compute.
    const float d2p = point[axis] - med;
    const bool anchor_sign = dax[indices[anchor + 1] * N] >= med;
    const bool search_rhs = (d2p >= 0) == anchor_sign;

    Scalar& lim_lhs = (anchor_sign ? pmax : pmin)[axis];
    Scalar& lim_rhs = (anchor_sign ? pmin : pmax)[axis];

    Scalar& lim_top = search_rhs ? lim_rhs : lim_lhs;
    Scalar& lim_bot = search_rhs ? lim_lhs : lim_rhs;

    // Update bounds and search first child.
    std::swap(lim_top, med);
    SearchNearestNeighborRecursiveLevel(point, k,
                                        anchor + (search_rhs ? step : -step),
                                        level + 1, pmin, pmax, q);
    std::swap(lim_top, med);

    // Update bounds and search second child.
    std::swap(lim_bot, med);
    SearchNearestNeighborRecursiveLevel(point, k,
                                        anchor + (search_rhs ? -step : step),
                                        level + 1, pmin, pmax, q);
    std::swap(lim_bot, med);
  }

  void SearchNearestNeighborRecursive(const Scalar* const point, int k,
                                      std::vector<int>* const out,
                                      const bool sort = false) const {
    // Priority queue of neighbors.
    TopKV2 q(k);

    // pmin and pmax are not actually modified, safe to const cast.
    // NOTE(yycho0108): May want to NOT do this if threading.
    SearchNearestNeighborRecursiveLevel(
        point, k, size >> 1, 0, const_cast<std::array<Scalar, N>&>(pmin),
        const_cast<std::array<Scalar, N>&>(pmax), &q);

    // Export output.
    out->resize(k);
    if (sort) {
      std::sort_heap(q.values.begin(), q.values.end());
    }
    std::transform(q.values.begin(), q.values.end(), out->begin(),
                   [](const auto& v) { return v.second; });
  }

  void SearchNearestNeighbor(const Scalar* const point, int k,
                             std::vector<int>* const out) const {
    // Maintain best k entries.
    TopKV2 q(k);
    std::stack<int> anchors;
    const int ax0 = ffs(size >> 1);
    anchors.emplace(size >> 1);  // midpoint
    bool ge_anchor = true;

    std::unordered_map<int, float> d_prv;
    d_prv[0] = 0;

    while (!anchors.empty()) {
      // Get element - DFS, extract from back
      auto anchor = anchors.top();
      anchors.pop();

      // Determine separating plane.
      const int fsb = ffs(anchor);
      const int level = ax0 - fsb;
      const int axis = level % N;

      // Determine which side of the anchor the point belongs.
      // Compute distance to hyperplane.
      const Scalar* const dax = data + axis;
      const Scalar d2p = point[axis] - dax[indices[anchor] * N];
      const bool anchor_sign =
          dax[indices[anchor + 1] * N] >= dax[indices[anchor] * N];
      const bool search_rhs = (d2p >= 0) == anchor_sign;

      // Squared Distance to the separating hyperplane.
      // Should technically be d_prv + (d2p * d2p),
      // where d_prv is the accumulated distance to the hyperplane
      // along each parent nodes.
      const Scalar d2p1 = d2p * d2p;

      // If leaf node, traverse through points and compare.
      const int i00 = search_rhs ? 0 : -leaf_size;
      const int i01 = search_rhs ? +leaf_size : 0;
      const int i10 = search_rhs ? -leaf_size : 0;
      const int i11 = search_rhs ? 0 : +leaf_size;
      if (anchor & leaf_size) {
        SearchLeaf(point, anchor + i00, anchor + i01, &q);
        // Other side of hyperplane is only evaluated if needed.
        if (q.size() < k || d2p1 < q.top().first) {
          SearchLeaf(point, anchor + i10, anchor + i11, &q);
        }
        continue;
      }

      // Otherwise, propagate to subtree.
      const int step = 1 << (fsb - 2);
      // Other side of hyperplane is only evaluated if needed.
      if (q.size() < k || d2p1 < q.top().first) {
        anchors.emplace(anchor + (search_rhs ? -step : step));
      }
      // since dfs, the last one is evaluated first.
      anchors.emplace(anchor + (search_rhs ? step : -step));
    }

    // Export output.
    out->resize(k);
    std::sort_heap(q.values.begin(), q.values.end());
    std::transform(q.values.begin(), q.values.end(), out->begin(),
                   [](const auto& v) { return v.second; });
  }

  KdTree(Scalar* const data, int size, int leaf_size = 16)
      : data(data), size(size), leaf_size(leaf_size) {
    // Track indices ...
    indices.resize(size);
    std::iota(indices.begin(), indices.end(), 0);

    depth = std::ceil(std::log2((size + leaf_size - 1) / leaf_size));
    // Actual leaf size, aligned to powers of 2.
    leaf_size = (size >> depth);
    for (int i = 0; i < depth; ++i) {
      SplitLevel(i, &indices);
    }

    ComputeExtents(&pmin, &pmax);
  }

  void ComputeExtents(std::array<Scalar, N>* const pmin,
                      std::array<Scalar, N>* const pmax) const {
    std::fill(pmin->begin(), pmin->end(), std::numeric_limits<Scalar>::max());
    std::fill(pmax->begin(), pmax->end(),
              std::numeric_limits<Scalar>::lowest());
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < N; ++j) {
        (*pmin)[j] = std::min((*pmin)[j], data[i * N + j]);
        (*pmax)[j] = std::max((*pmax)[j], data[i * N + j]);
      }
    }
  }

  Scalar* data;
  int size;
  int depth;
  int leaf_size;
  std::vector<int> indices;
  std::array<Scalar, N> pmin, pmax;
};

template <int kDim = 2>
void DrawKdTreeAtLevel(const KdTree<float, kDim>& tree, cv::Mat* const img,
                       const int level, int i0, int i1,
                       std::array<float, kDim>& pmin,
                       std::array<float, kDim>& pmax,
                       const std::function<cv::Point2f(float, float)>& pmap) {
  // if (level >= 5) {
  //  return;
  //}
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
    // fmt::print("Anchor2 @ {} = ({},{})\n", im, x, y);
    float median = tree.data[indices[im] * kDim + axis];

    if (tree.data[indices[i0] * kDim + axis] <
        tree.data[indices[im] * kDim + axis]) {
      std::swap(pmax[axis], median);
      DrawKdTreeAtLevel<kDim>(tree, img, level + 1, i0, im, pmin, pmax, pmap);
      std::swap(pmax[axis], median);

      std::swap(pmin[axis], median);
      DrawKdTreeAtLevel<kDim>(tree, img, level + 1, im, i1, pmin, pmax, pmap);
      std::swap(pmin[axis], median);
    } else {
      std::swap(pmin[axis], median);
      DrawKdTreeAtLevel<kDim>(tree, img, level + 1, i0, im, pmin, pmax, pmap);
      std::swap(pmin[axis], median);

      std::swap(pmax[axis], median);
      DrawKdTreeAtLevel<kDim>(tree, img, level + 1, im, i1, pmin, pmax, pmap);
      std::swap(pmax[axis], median);
    }

    // Draw the splitting line.
    {
      float mn = pmin[axis];
      float mx = pmax[axis];
      pmin[axis] = median;
      pmax[axis] = median;
      cv::line(
          *img, pmap(pmin[0], pmin[1]), pmap(pmax[0], pmax[1]),
          cv::Scalar((level % 3 == 2) ? 255 : 128, (level % 3 == 1) ? 255 : 128,
                     (level % 3 == 0) ? 255 : 128));
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

template <int kDim = 2>
void DrawKdTree2d(const KdTree<float, kDim>& tree, cv::Mat* const img) {
  // static constexpr const int kDim = 2;
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

  DrawKdTreeAtLevel<kDim>(
      tree, img, 0, 0, static_cast<int>(tree.indices.size()), pmin, pmax, pmap);
  return;
}

int main() {
  constexpr const int kNumPoints(256);
  constexpr const int kDim = 3;

  constexpr const int kNumIter = 64;
  constexpr const int kNeighbors = 32;

  using KdTreeIndex = PointerAdaptor<float, kDim>;
  using KdTreeNanoflann = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, KdTreeIndex>, KdTreeIndex, kDim>;

  std::vector<float> dat(kNumPoints * kDim);

  // Fill with random data
  auto& rng = cho_util::core::RNG::GetInstance();
  // const auto seed =
  // std::chrono::high_resolution_clock::now().time_since_epoch().count();
  // const auto seed = 1586637242494121116;
  const auto seed = 1586689800866777869;
  fmt::print("Seed = {}\n", seed);
  rng.SetSeed(seed);
  std::generate(dat.begin(), dat.end(), [&rng]() { return rng.Randu(); });

  // nanoflann
  KdTreeIndex index_(dat.data(), kNumPoints);
  KdTreeNanoflann tree2(kDim, index_);
  tree2.buildIndex();
  std::vector<std::size_t> oi(kNeighbors);
  std::vector<float> od(kNeighbors);

  // me
  KdTree<float, kDim> tree(dat.data(), kNumPoints, 16);
  std::vector<int> out(kNeighbors);

  {
    int dummy{0};
    cho_util::util::MTimer timer{true};
    for (int m = 0; m < kNumIter; ++m) {
      for (int i = 0; i < kNumPoints; ++i) {
        tree.SearchNearestNeighborRecursive(&dat[i * kDim], kNeighbors, &out,
                                            false);
        // tree.SearchNearestNeighbor(&dat[i * kDim], kNeighbors, &out);
        dummy += out.back();

        // if (nbr == 0) {
        //  std::cout << m << std::endl;
        //  throw std::runtime_error("nbr0");
        //}
      }
    }
    fmt::print("?{}\n", dat[kNumPoints - 1 * kDim]);
    fmt::print("me    {} ms\n", timer.StopAndGetElapsedTime());
    fmt::print("dummy {}\n", dummy);
  }

  {
    int dummy{0};
    cho_util::util::MTimer timer{true};
    CALLGRIND_TOGGLE_COLLECT;
    for (int m = 0; m < kNumIter; ++m) {
      for (int i = 0; i < kNumPoints; ++i) {
        tree2.knnSearch(&dat[i * kDim], kNeighbors, oi.data(), od.data());
        dummy += oi.back();
      }
    }
    CALLGRIND_TOGGLE_COLLECT;

    fmt::print("?{}\n", dat[kNumPoints - 1 * kDim]);
    fmt::print("nano  {} ms\n", timer.StopAndGetElapsedTime());
    fmt::print("dummy {}\n", dummy);
  }

  fmt::print("me        {}\n", fmt::join(out, " "));
  fmt::print("nanoflann {}\n", fmt::join(oi, " "));

  // for (int i = 0; i < kDim; ++i) {
  //  fmt::print("{} {}\n", dat[kTestIndex * kDim + i], test[i]);
  //}

  // fmt::print("{}\n", nbr);

  if (kDim == 2) {
    cv::Mat img(512, 512, CV_8UC3);
    DrawKdTree2d(tree, &img);
    cv::imshow("tree", img);
    cv::imwrite("/tmp/kdtree.png", img);
    cv::waitKey(0);
  } else if (kDim == 3) {
    namespace cuv = cho_util::vis;
    cuv::EmbeddedViewer v{true};
    // Render points
    {
      cuv::RenderData rd;
      rd.render_type = cuv::RenderData::RenderType::kPoints;
      rd.data = dat;
      rd.tag = "points";
      rd.quit = false;
      v.Render(rd);
    }

    // Render boxes
    {
      cuv::RenderData rd;
      rd.render_type = cuv::RenderData::RenderType::kBox;
      rd.quit = false;
      int count = 0;
      rd.data.resize(6);
      rd.color.resize(3);
      std::this_thread::sleep_for(std::chrono::milliseconds(10000));
      tree.VisitBoxes([&rd, &v, &count, &rng](
                          const std::array<float, kDim>& bmin,
                          const std::array<float, kDim>& bmax) {
        std::copy(bmin.begin(), bmin.end(), rd.data.begin());
        std::copy(bmax.begin(), bmax.end(), rd.data.begin() + kDim);
        std::generate(rd.color.begin(), rd.color.end(),
                      [&rng]() -> std::uint8_t { return rng.Uniform(0, 255); });

        rd.tag = fmt::format("box-{}", count);
        // rd.tag = fmt::format("box", count);

        rd.representation = cuv::RenderData::Representation::kSurface;
        v.Render(rd);

        rd.tag += "-edge";
        rd.representation = cuv::RenderData::Representation::kWireframe;
        v.Render(rd);

        ++count;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      });
    }

    // wait
    std::this_thread::sleep_for(std::chrono::seconds(100));
  }
  // for (const auto& i : tree.indices) {
  //  fmt::print("{}\n", i);
  //}

  // Draw ...
}
