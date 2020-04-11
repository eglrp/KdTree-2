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

#include <valgrind/callgrind.h>

#include "heap.hpp"

// #define NOTOPK

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
      values.emplace_back(std::move(value));
      if (values.size() == size_) {
        std::make_heap(values.begin(), values.end());
      }
    } else {
      if (value >= values.front()) {
        return;
      }
      values.front() = std::move(value);
      heapreplace(values.begin(), values.end());
    }
  }

  inline void Add(const std::pair<float, int>& value) {
    if (values.size() < size_) {
      values.emplace_back(value);
      if (values.size() == size_) {
        std::make_heap(values.begin(), values.end());
      }
    } else {
      if (value >= values.front()) {
        return;
      }
      values.front() = value;
      heapreplace(values.begin(), values.end());
    }
  }

  inline int size() const { return values.size(); }
  inline const std::pair<float, int>& top() const { return values.front(); }
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
      const float d = a[i] - b[i];
      out += d * d;
      // out += (a[i] - b[i]) * (a[i] - b[i]);
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

      // TODO(yycho0108): Non-hacky way to deal with skipping higher-level
      // anchors.
      // fmt::print("{} ax={}\n", std::distance(indices->begin(), im), axis);
      std::nth_element(i0 + (i0 != indices->begin()), im, i1,
                       [dax, &axis](const int lhs, const int rhs) {
                         return dax[lhs * N] < dax[rhs * N];
                       });
    }
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

  inline int ParentIndex(const int child) const {
    const int delta = ffs(child);
    return (child >> delta) << delta;
  }

  inline bool SearchLeaf(
      const Scalar* const point, const int i, int k,
#ifdef NOTOPK
      std::priority_queue<std::pair<float, int>,
                          std::vector<std::pair<float, int>>>* const q) const {
#else
      TopKV2* const q) const {
#endif
    const int index = indices[i];
    const Scalar d = Distance(&data[index * N], point);

    // Insert element to priority queue.
#ifdef NOTOPK
    if (q->size() < k || d < q->top().first) {
      if (q->size() == k) {
        q->pop();
      }
      q->emplace(d, index);
    }
#else
    q->Add({d, i});
#endif
  }

  inline bool SearchLeaf(
      const Scalar* const point, const int imin, const int imax, int k,
#ifdef NOTOPK
      std::priority_queue<std::pair<float, int>,
                          std::vector<std::pair<float, int>>>* const q) const {
#else
      TopKV2* const q) const {
#endif
    // Search leaf.
    // fmt::print("{} - {}\n", imin, imax);
    for (int i = imin; i < imax; ++i) {
      // SearchLeaf(point, i, k, q);
      const int index = indices[i];
      const Scalar d = Distance(&data[index * N], point);

      // Insert element to priority queue.
#ifdef NOTOPK
      if (q->size() < k || d < q->top().first) {
        if (q->size() == k) {
          q->pop();
        }
        q->emplace(d, index);
      }
#else
      q->Add({d, i});
#endif
    }
  }

  int SearchNearestNeighbor(const Scalar* const point, int k,
                            std::vector<int>* const out) const {
    // Maintain best k entries.
#ifdef NOTOPK
    std::vector<std::pair<float, int>> container;
    container.reserve(k);
    const auto ptr = container.data();
    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int>>>
    q(std::less<std::pair<float, int>>(), std::move(container));
#else
    TopKV2 q(k);
#endif

    // Search exact location for debugging ...
    // std::priority_queue<std::pair<float, int>> dbgq;
    // for (int i = 0; i < size; ++i) {
    //  const float d = Distance(&data[indices[i] * N], point);
    //  if (dbgq.size() < k || d < dbgq.top().first) {
    //    if (dbgq.size() == k) {
    //      dbgq.pop();
    //    }
    //    dbgq.emplace(d, indices[i]);
    //  }
    //}
    // visited.resize(indices.size(), false);
    {
      std::deque<int> anchors;
      const int ax0 = ffs(size >> 1);
      anchors.emplace_back(size >> 1);  // midpoint
      bool ge_anchor = true;

      // Determine if the other side should be searched.
      // auto expand = [&q, k](const float d) {
      //   return q.size() < k || d < q.top().first;
      // };

      while (!anchors.empty()) {
        // Get element - DFS
        auto anchor = anchors.back();
        anchors.pop_back();
        // fmt::print("anchor : {}\n", anchor);

        // Determine separating plane.
        const int fsb = ffs(anchor);
        const int level = ax0 - fsb;
        const int axis = level % N;
        // fmt::print("range : {} - {}\n", anchor-step*2, anchor+step*2);
        // fmt::print("anchor {} ax={}\n", anchor, axis);

        // Determine which side of the anchor the point belongs.

        // distance to hyperplane
        const Scalar d2p = point[axis] - data[indices[anchor] * N + axis];
        const bool ge_anchor = d2p >= 0;

        // Squared Distance to the separating hyperplane.
        const Scalar d = d2p * d2p;
        // if (q.size()) {
        // fmt::print("{} , {}\n", d, q.top().first);
        // }

        // Process leaf node.
        if (anchor & leaf_size) {
          if (ge_anchor) {
            // Evaluate right of anchor first.
            SearchLeaf(point, anchor, anchor + leaf_size, k, &q);

            // Then Evaluate left of anchor.
            if (q.size() < k || d < q.top().first) {
              SearchLeaf(point, anchor - leaf_size, anchor, k, &q);
            } else {
              // Evaluate parent node, whose hyperplane membership is
              // unclear.
              SearchLeaf(point, anchor - leaf_size, k, &q);
            }
          } else {
            // Evaluate left of anchor first.
            SearchLeaf(point, anchor - leaf_size, anchor, k, &q);
            // Then conditionally evaluate right of anchor.
            if (q.size() < k || d < q.top().first) {
              SearchLeaf(point, anchor, anchor + leaf_size, k, &q);
            }
          }
          continue;
        }

        // Otherwise, propagate.
        const int step = 1 << (fsb - 2);
        if (ge_anchor) {
          // Other side of hyperplane is only evaluated if needed.
          if (q.size() < k || d < q.top().first) {
            anchors.emplace_back(anchor - step);
          } else {
            // Parent node still needs to be searched, hyperplane not guaranteed
            if (anchor - 2 * step >= 0) {
              SearchLeaf(point, anchor - 2 * step, k, &q);
            }
          }
          anchors.emplace_back(anchor + step);
        } else {
          // Other side of hyperplane is only evaluated if needed.
          if (q.size() < k || d < q.top().first) {
            anchors.emplace_back(anchor + step);
          }
          anchors.emplace_back(anchor - step);
        }
      }
      // fmt::print("Done\n");
      // if (q.top() != dbgq.top()) {
      //  fmt::print("dbgq {}\n", dbgq);
      //  fmt::print("q {}\n", q);
      //  return false;
      //}

      // export
#ifdef NOTOPK
      out->resize(k);
      std::transform(ptr, ptr + k, out->begin(),
                     [](const auto& x) { return x.second; });
      // std::copy(ptr, ptr + k, out->begin());
      // for (int i = k - 1; i >= 0; --i) {
      //  out->at(i) = q.top().second;
      //  q.pop();
      //}
#else
      out->clear();
      out->reserve(k);
      std::sort_heap(q.values.begin(), q.values.end());
      for (const auto& v : q.values) {
        out->emplace_back(v.second);
      }
#endif
    }

    return 1;
#if 0
    // Search node membership.
    int anchor = 0;
    bool ge_anchor = true;
    for (int i = 0; i < depth; ++i) {
      // 256, 256+{}
      anchor += (ge_anchor ? 1 : -1) * (size >> (i + 1));
      const int axis = i % N;
      ge_anchor = point[axis] >= data[indices[anchor] * N + axis];
    }

    bool top_level = depth;
    while (true) {
      // Search leaf.
      int imin = (ge_anchor) ? anchor : anchor - leaf_size;
      int imax = (ge_anchor) ? anchor + leaf_size : anchor;
      for (auto it = indices.begin() + imin; it != indices.begin() + imax;
           ++it) {
        const auto& index = *it;
        const Scalar d = Distance(&data[index * N], point);

        // Insert element to priority queue.
        if (q.size() < k || d < q.top().first) {
          if (q.size() == k) {
            q.pop();
          }
          q.emplace(Distance(&data[index * N], point), index);
        }
      }

      // Compute distance to separating plane.
      const int axis = (depth - 1) % N;
      const Scalar d =
          (ge_anchor ? -1 : 1) * (point[axis] - data[anchor * N + axis]);

      // Done if full and all other points are farther.
      if (q.size() == k && d >= q.top().first) {
        break;
      }

      // Determine next search target ...
    }

    // fmt::print("BestD : {}\n", q.top().first);
    // const int axis = (depth - 1) % N;
    // const Scalar d =
    //    (ge_anchor ? -1 : 1) * (point[axis] - data[anchor * N + axis]);
    // fmt::print("Distance to plane : {}\n", d);

    // fmt::print("{} {}\n", imin, imax);

    // const auto index =
    //    *std::min_element(indices.begin() + imin, indices.begin() + imax,
    //                      [this, &point](const int lhs, const int rhs) {
    //                        return Distance(&data[lhs * N], point) <
    //                               Distance(&data[rhs * N], point);
    //                      });
    //// output
    // for (int i = 0; i < N; ++i) {
    //  out[i] = data[index * N + i];
    //}
#endif
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
  }

  Scalar* data;
  int size;
  int depth;
  int leaf_size;
  std::vector<int> indices;
  // mutable std::vector<bool> visited;
};

void DrawKdTreeAtLevel(const KdTree<float, 2>& tree, cv::Mat* const img,
                       const int level, int i0, int i1,
                       std::array<float, 2>& pmin, std::array<float, 2>& pmax,
                       const std::function<cv::Point2f(float, float)>& pmap) {
  //if (level >= 3) {
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
    // fmt::print("Anchor2 @ {} = ({},{})\n", im, x, y);
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
  constexpr const int kNumPoints(1024);
  constexpr const int kDim = 2;

  constexpr const int kNumIter = 128;
  constexpr const int kNeighbors = 17;

  using KdTreeIndex = PointerAdaptor<float, kDim>;
  using KdTreeNanoflann = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, KdTreeIndex>, KdTreeIndex, kDim>;

  std::vector<float> dat(kNumPoints * kDim);

  // Fill with random data
  auto& rng = cho_util::core::RNG::GetInstance();
  const auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  // const auto seed = 1586637242494121116;
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
  KdTree<float, kDim> tree(dat.data(), kNumPoints, 8);
  std::vector<int> out(kNeighbors);

  {
    int dummy{0};
    cho_util::util::MTimer timer{true};
    CALLGRIND_TOGGLE_COLLECT;
    for (int m = 0; m < kNumIter; ++m) {
      for (int i = 0; i < kNumPoints; ++i) {
        tree.SearchNearestNeighbor(&dat[i * kDim], kNeighbors, &out);
        dummy += out.back();

        // if (nbr == 0) {
        //  std::cout << m << std::endl;
        //  throw std::runtime_error("nbr0");
        //}
      }
    }
    CALLGRIND_TOGGLE_COLLECT;
    fmt::print("me    {} ms\n", timer.StopAndGetElapsedTime());
    fmt::print("dummy {}\n", dummy);
  }

  {
    int dummy{0};
    cho_util::util::MTimer timer{true};
    for (int m = 0; m < kNumIter; ++m) {
      for (int i = 0; i < kNumPoints; ++i) {
        tree2.knnSearch(&dat[i * kDim], kNeighbors, oi.data(), od.data());
        dummy += oi.back();
      }
    }
    fmt::print("nano  {} ms\n", timer.StopAndGetElapsedTime());
    fmt::print("dummy {}\n", dummy);
  }

  fmt::print("me        {}\n", fmt::join(out, " "));
  fmt::print("nanoflann {}\n", fmt::join(oi, " "));

  // for (int i = 0; i < kDim; ++i) {
  //  fmt::print("{} {}\n", dat[kTestIndex * kDim + i], test[i]);
  //}

  // fmt::print("{}\n", nbr);
  cv::Mat img(512, 512, CV_8UC1);
  DrawKdTree2d(tree, &img);
  cv::imshow("tree", img);
  cv::imwrite("/tmp/kdtree.png", img);
  cv::waitKey(0);
  // for (const auto& i : tree.indices) {
  //  fmt::print("{}\n", i);
  //}

  // Draw ...
}
