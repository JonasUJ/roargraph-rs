use crate::adj_list_graph::AdjListGraph;
use hnsw_itu::{Distance, Index, IndexVis, Point};
use min_max_heap::MinMaxHeap;
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};
use tracing::info;

pub fn select_neighbors<'a, P: Point>(
    mut candidates: MinMaxHeap<Distance<'a, P>>,
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

pub fn select_neighbors_max<'a, T: Point>(
    mut candidates: MinMaxHeap<Distance<'a, T>>,
    m: usize,
) -> Vec<Distance<'a, T>> {
    if candidates.len() <= m {
        return candidates.drain_asc().take(m).collect();
    }

    let mut return_list = Vec::<Distance<'a, T>>::new();
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

#[derive(Serialize, Deserialize)]
pub struct RoarGraph<T> {
    medoid: usize,
    pub graph: AdjListGraph<T>,
}

impl<P: Point> Index<P> for RoarGraph<P> {
    type Options<'a> = usize;
    fn size(&self) -> usize {
        self.graph.size()
    }
    fn search(&'_ self, query: &P, k: usize, ef: &Self::Options<'_>) -> Vec<Distance<'_, P>> {
        let mut res = self.graph.search(query, *ef, &self.medoid);
        res.truncate(k);
        res
    }
}

impl<P: Point> IndexVis<P> for RoarGraph<P> {
    fn search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        ef: &Self::Options<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>> {
        let mut res = self.graph.search_vis(query, *ef, &self.medoid, vis);
        res.truncate(k);
        res
    }
}

impl<P: Point> Index<P> for AdjListGraph<P> {
    type Options<'a> = usize;

    fn size(&self) -> usize {
        self.size()
    }

    fn search(&'_ self, query: &P, k: usize, medoid: &Self::Options<'_>) -> Vec<Distance<'_, P>> {
        self.search_vis(query, k, medoid, &mut HashSet::new())
    }
}

impl<P: Point> IndexVis<P> for AdjListGraph<P> {
    fn search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        medoid: &Self::Options<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>> {
        let medoid_element = self.get(*medoid).expect("entry point was not in graph");
        let query_distance = Distance::new(medoid_element.distance(query), *medoid, medoid_element);

        let mut visited = HashSet::with_capacity(2048);
        visited.insert(query_distance.clone());
        let mut w = MinMaxHeap::from_iter([query_distance.clone()]);
        let mut candidates = MinMaxHeap::from_iter([query_distance]);

        while !candidates.is_empty() {
            let c = candidates.pop_min().expect("candidates can't be empty");
            let f = w.peek_max().expect("w can't be empty");

            if c.distance > f.distance {
                break;
            }

            for e in self.neighborhood(c.key) {
                let point = self.get(e).unwrap();
                if visited.contains(&Distance::new(0.0, e, point)) {
                    continue;
                }

                let f = w.peek_max().expect("w can't be empty");

                let e_dist = Distance::new(point.distance(query), e, point);
                visited.insert(e_dist.clone());

                if e_dist.distance >= f.distance && w.len() >= k {
                    continue;
                }

                candidates.push(e_dist.clone());
                w.push(e_dist);

                if w.len() > k {
                    w.pop_max();
                }
            }
        }

        *vis = visited;

        w.drain_asc().collect()
    }
}

#[derive(Debug)]
pub struct RoarGraphOptions {
    /// Out-degree bound
    pub m: usize,
    /// Candidate pool size
    pub l: usize,
}

pub struct RoarGraphBuilder {
    options: RoarGraphOptions,
    frequency_map: Vec<usize>,
}

impl RoarGraphBuilder {
    pub fn new(options: RoarGraphOptions) -> Self {
        Self {
            options,
            frequency_map: Vec::new(),
        }
    }

    pub fn build<P: Point + Clone + Send + Sync>(
        mut self,
        queries: Vec<P>,
        data: Vec<P>,
        ground_truth: Vec<Vec<usize>>,
    ) -> RoarGraph<P> {
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

        // Bipartite projection
        info!("Projecting bipartite graph...");
        let mut projected_graph =
            self.neighborhood_aware_projection(bipartite_graph, data.iter().cloned().collect());

        info!("Computing medoid...");
        //let medoid = 78861; //im;
        //let medoid = 213373; //ll;
        let mut rng = rand::rng();
        let sample = projected_graph
            .adj_lists()
            .iter()
            .flat_map(|m| m)
            .sample(&mut rng, queries.len() / 10)
            .into_iter()
            .collect::<HashSet<_>>();
        let medoid = sample
            .par_iter()
            .map(|&&p| {
                let point = projected_graph.get(p).expect("point to be in graph");
                let total_distance: f32 = sample
                    .iter()
                    .map(|&&o| {
                        let other = projected_graph.get(o).expect("point to be in graph");
                        point.distance(other)
                    })
                    .sum();
                (total_distance, p)
            })
            .min_by(|(dist_a, _), (dist_b, _)| OrderedFloat(*dist_a).cmp(&OrderedFloat(*dist_b)))
            .map(|(_, i)| i)
            .expect("data set is empty");
        info!("Medoid index: {}", medoid);

        // Connectivity enhancement
        info!("Enhancing connectivity...");
        let all_candidates = data
            .par_iter()
            .map(|p| projected_graph.search(p, self.options.m, &medoid))
            .enumerate()
            .collect::<Vec<_>>();
        let mut conn_graph = projected_graph.clone();
        for (i, candidates) in all_candidates {
            let selected_neighbors = select_neighbors(candidates.into(), self.options.m);
            conn_graph.set_neighbors(
                i,
                selected_neighbors.iter().map(|d: &Distance<'_, P>| d.key),
            );

            for p in selected_neighbors {
                let p_candidates = MinMaxHeap::from_iter(
                    conn_graph
                        .neighborhood(p.key)
                        .chain(std::iter::once(i))
                        .map(|n| {
                            let neighbor_point =
                                conn_graph.get(n).expect("point not found in graph");
                            Distance::new(p.point.distance(neighbor_point), n, neighbor_point)
                        }),
                );
                let p_neighbors: Vec<usize> = select_neighbors(p_candidates, self.options.m)
                    .into_iter()
                    .map(|d| d.key)
                    .collect();
                conn_graph.set_neighbors(p.key, p_neighbors.into_iter());
            }
        }

        for i in 0..projected_graph.nodes().len() {
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

    fn neighborhood_aware_projection<P: Point>(
        &self,
        bipartite_graph: AdjListGraph<&P>,
        data: Vec<P>,
    ) -> AdjListGraph<P> {
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

                        if candidates.len() >= self.options.l {
                            break 'outer;
                        }
                    }
                }
            }

            let selected_neighbors = select_neighbors_max(candidates, self.options.m);
            let selected_neighbors = selected_neighbors.iter().map(|d| d.key).collect::<Vec<_>>();
            projected_graph.set_neighbors(x, selected_neighbors.iter().copied());

            for p in selected_neighbors {
                let point = projected_graph.get(p).expect("point to be in graph");
                let new_candidates = MinMaxHeap::from_iter(
                    projected_graph.neighborhood(p).chain(vec![x]).map(|n| {
                        let neighbor_point = projected_graph.get(n).expect("point to be in graph");
                        Distance::new(point.distance(neighbor_point), n, neighbor_point)
                    }),
                );
                let selected = select_neighbors_max(new_candidates, self.options.m)
                    .iter()
                    .map(|d| d.key)
                    .collect::<Vec<_>>();
                projected_graph.set_neighbors(p, selected.into_iter());
            }
        }

        projected_graph
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
