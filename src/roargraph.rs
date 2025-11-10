use crate::adj_list_graph::AdjListGraph;
use crate::point::{Distance, Point};
use min_max_heap::MinMaxHeap;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet};
use tracing::info;

// Heuristic
pub(crate) fn select_neighbors<'a, P: Point>(
    candidates: &mut MinMaxHeap<Distance<'a, P>>,
    m: usize,
) -> Vec<Distance<'a, P>> {
    let mut return_list = Vec::<Distance<'a, P>>::new();

    while let Some(e) = candidates.pop_min() {
        if return_list.len() >= m {
            break;
        }

        if return_list
            .iter()
            .all(|r| e.point.distance(r.point) > e.distance)
        {
            return_list.push(e);
        }
    }

    return_list
}

pub(crate) fn select_neighbors_max<'a, P: Point>(
    candidates: &mut MinMaxHeap<Distance<'a, P>>,
    m: usize,
) -> Vec<Distance<'a, P>> {
    let mut return_list = Vec::<Distance<'a, P>>::new();
    let mut rejects = MinMaxHeap::new();

    while let Some(e) = candidates.pop_min() {
        if return_list.len() >= m {
            break;
        }

        if return_list
            .iter()
            .all(|r| e.point.distance(r.point) > e.distance)
        {
            return_list.push(e);
        } else {
            rejects.push(e);
        }
    }

    return_list.extend(rejects.drain_asc().take(m - return_list.len()));
    return_list
}

pub(crate) fn neighborhood_aware_projection<T: Point>(
    bipartite_graph: AdjListGraph<&T>,
    data: Vec<T>,
    limit: usize,
    candidate_size: usize,
) -> AdjListGraph<T> {
    let mut projected_graph = AdjListGraph::with_nodes(data);

    for x in 0..projected_graph.size() {
        let mut out_neighbors = bipartite_graph.neighborhood(x).into_iter().peekable();
        if out_neighbors.peek().is_none() {
            continue;
        }

        let x_point = projected_graph.get(x).expect("point to be in graph");
        let mut candidates = MinMaxHeap::new();
        'outer: for s in out_neighbors {
            for neighbor in bipartite_graph.neighborhood(s) {
                if neighbor != x {
                    let neighbor_point =
                        projected_graph.get(neighbor).expect("point to be in graph");
                    candidates.push(Distance::new(
                        x_point.distance(neighbor_point),
                        neighbor,
                        neighbor_point,
                    ));

                    if candidates.len() >= candidate_size {
                        break 'outer;
                    }
                }
            }
        }

        let selected_neighbors = select_neighbors_max(&mut candidates, limit);
        let selected_neighbors = selected_neighbors.iter().map(|d| d.key).collect::<Vec<_>>();
        projected_graph.set_neighbors(x, selected_neighbors.iter().copied());

        for p in selected_neighbors {
            let point = projected_graph.get(p).expect("point to be in graph");
            let mut new_candidates =
                MinMaxHeap::from_iter(projected_graph.neighborhood(p).chain(vec![x]).map(|n| {
                    let neighbor_point = projected_graph.get(n).expect("point to be in graph");
                    Distance::new(point.distance(neighbor_point), n, neighbor_point)
                }));
            let selected = select_neighbors_max(&mut new_candidates, limit)
                .iter()
                .map(|d| d.key)
                .collect::<Vec<_>>();
            projected_graph.set_neighbors(p, selected.into_iter());
        }
    }

    projected_graph
}

pub struct RoarGraph<T> {
    medoid: usize,
    graph: AdjListGraph<T>,
}

impl<T> RoarGraph<T> {
    fn new() -> Self {
        Self {
            medoid: 0,
            graph: AdjListGraph::new(),
        }
    }
}

impl<P: Point> RoarGraph<P> {
    pub fn search(&'_ self, query: &P, ef: usize) -> MinMaxHeap<Distance<'_, P>> {
        self.graph.search(query, self.medoid, ef)
    }
}

impl<P: Point> AdjListGraph<P> {
    fn search(&'_ self, query: &P, medoid: usize, ef: usize) -> MinMaxHeap<Distance<'_, P>> {
        let medoid_element = self.get(medoid).expect("entry point was not in graph");
        let query_distance = Distance::new(medoid_element.distance(query), medoid, medoid_element);

        let mut visited = HashSet::with_capacity(2048);
        visited.insert(medoid);
        let mut w = MinMaxHeap::from_iter([query_distance.clone()]);
        let mut candidates = MinMaxHeap::from_iter([query_distance]);

        while !candidates.is_empty() {
            let c = candidates.pop_min().expect("candidates can't be empty");
            let f = w.peek_max().expect("w can't be empty");

            if c.distance > f.distance {
                break;
            }

            for e in self.neighborhood(c.key) {
                if visited.contains(&e) {
                    continue;
                }

                visited.insert(e);
                let f = w.peek_max().expect("w can't be empty");

                let point = self.get(e).unwrap();
                let e_dist = Distance::new(point.distance(query), e, point);

                if e_dist.distance >= f.distance && w.len() >= ef {
                    continue;
                }

                candidates.push(e_dist.clone());
                w.push(e_dist);

                if w.len() > ef {
                    w.pop_max();
                }
            }
        }

        w
    }
}

pub struct RoarGraphOptions {
    /// Number of nearest neighbors to consider for each query
    pub(crate) nq: usize,
    /// Out-degree bound
    pub(crate) m: usize,
    /// Candidate pool size
    pub(crate) l: usize,
}

pub struct RoarGraphBuilder {
    options: RoarGraphOptions,
}

impl RoarGraphBuilder {
    pub fn new(options: RoarGraphOptions) -> Self {
        Self { options }
    }

    pub fn build<T: Point + Clone + Send + Sync>(
        &self,
        queries: Vec<T>,
        data: Vec<T>,
        ground_truth: Vec<Vec<usize>>,
    ) -> RoarGraph<T> {
        // Construct bipartite graph
        info!("Constructing bipartite graph...");
        let mut bipartite_graph = AdjListGraph::with_nodes(data.iter().chain(&queries).collect());
        for (v, closest) in ground_truth.iter().enumerate() {
            let v = v + data.len();
            if let Some(&w) = closest.first() {
                bipartite_graph.add_directed_edge(w, v)
            }
            for &w in closest.iter().skip(1) {
                bipartite_graph.add_directed_edge(v, w);
            }
        }

        info!("Computing medoid...");
        let medoid = 184379; //7331;
        //let sample = bipartite_graph
        //    .adj_lists()
        //    .iter()
        //    .skip(data.len())
        //    .flat_map(|m| m)
        //    .collect::<HashSet<_>>();
        //let medoid = sample
        //    .par_iter()
        //    .map(|&&p| {
        //        let point = bipartite_graph.get(p).expect("point to be in graph");
        //        let total_distance: T::DistanceMetric = sample
        //            .iter()
        //            .map(|&&o| {
        //                let other = bipartite_graph.get(o).expect("point to be in graph");
        //                point.distance(other)
        //            })
        //            .sum();
        //        (total_distance, p)
        //    })
        //    .min_by(|(dist_a, _), (dist_b, _)| dist_a.cmp(dist_b))
        //    .map(|(_, i)| i)
        //    .expect("data set is empty");
        info!("Medoid index: {}", medoid);

        // Bipartite projection
        info!("Projecting bipartite graph...");
        let mut projected_graph = neighborhood_aware_projection(
            bipartite_graph,
            data.iter().cloned().collect(),
            self.options.nq,
            self.options.l,
        );

        // Connectivity enhancement
        info!("Enhancing connectivity...");
        let all_candidates = data
            .par_iter()
            .map(|p| projected_graph.search(p, medoid, self.options.m))
            .enumerate()
            .collect::<Vec<_>>();
        let mut conn_graph = projected_graph.clone();
        for (i, mut candidates) in all_candidates {
            let selected_neighbors = select_neighbors(&mut candidates, self.options.m);
            conn_graph.set_neighbors(i, selected_neighbors.iter().map(|d| d.key));

            for p in selected_neighbors {
                let mut p_candidates =
                    MinMaxHeap::from_iter(conn_graph.neighborhood(p.key).chain(vec![i]).map(|n| {
                        let neighbor_point = conn_graph.get(n).expect("point not found in graph");
                        Distance::new(p.point.distance(neighbor_point), n, neighbor_point)
                    }));
                let p_neighbors: Vec<usize> = select_neighbors(&mut p_candidates, self.options.m)
                    .into_iter()
                    .map(|d| d.key)
                    .collect();
                conn_graph.set_neighbors(i, p_neighbors.into_iter());
            }
        }

        for i in 1..projected_graph.nodes().len() {
            let mut final_neighbors = projected_graph.neighborhood(i).collect::<HashSet<usize>>();
            final_neighbors.extend(conn_graph.neighborhood(i));
            projected_graph.set_neighbors(i, final_neighbors.into_iter());
        }

        info!("RoarGraph construction complete");
        RoarGraph {
            medoid,
            graph: projected_graph,
        }
    }
}

pub trait MinK: Iterator {
    fn min_k(mut self, k: usize) -> Vec<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        if k == 0 {
            return vec![];
        }

        let iter = self.by_ref();
        let mut heap: BinaryHeap<Self::Item> = iter.take(k).collect();

        for i in iter {
            let mut top = heap
                .peek_mut()
                .expect("k is greater than 0 but heap was emptied");

            if top.gt(&i) {
                *top = i;
            }
        }

        heap.into_vec()
    }
}

impl<T> MinK for T where T: Iterator {}
