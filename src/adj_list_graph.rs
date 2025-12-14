use std::collections::HashSet;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct AdjListGraph<T> {
    nodes: Vec<T>,
    adj_lists: Vec<HashSet<usize>>,
    empty: HashSet<usize>,
}

impl<T> AdjListGraph<T> {
    pub fn nodes(&self) -> &Vec<T> {
        &self.nodes
    }

    pub fn adj_lists(&self) -> &Vec<HashSet<usize>> {
        &self.adj_lists
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_nodes(nodes: Vec<T>) -> Self {
        let len = nodes.len();
        Self {
            nodes,
            adj_lists: vec![HashSet::default(); len],
            empty: Default::default(),
        }
    }

    fn is_in_bounds(&self, v: usize, w: usize) -> bool {
        let len = self.adj_lists.len();
        v < len && w < len
    }

    fn connect_directed(&mut self, src: usize, target: usize) {
        if let Some(set) = self.adj_lists.get_mut(src) {
            set.insert(target);
        } else {
            panic!(
                "Index out of bounds: the len is {} but the index is {}",
                self.adj_lists.len(),
                src
            );
        }
    }

    fn disconnect_directed(&mut self, src: usize, target: usize) {
        if let Some(set) = self.adj_lists.get_mut(src) {
            set.remove(&target);
        }
    }

    fn is_connected(&self, v: usize, w: usize) -> bool {
        self.neighborhood(v).any(|i| i == w)
    }

    fn degree(&self, v: usize) -> usize {
        self.neighborhood(v).count()
    }

    fn clear_edges(&mut self, v: usize) {
        let neighbors = self.neighborhood(v).collect::<Vec<_>>();
        for w in neighbors {
            self.remove_edge(v, w);
        }
    }

    fn add_edges(&mut self, edges: impl Iterator<Item = (usize, usize)>) {
        for (v, w) in edges {
            self.add_edge(v, w);
        }
    }

    fn add_directed_edges(&mut self, edges: impl Iterator<Item = (usize, usize)>) {
        for (v, w) in edges {
            self.add_directed_edge(v, w);
        }
    }

    fn add_neighbors(&mut self, v: usize, neighbors: impl Iterator<Item = usize>) {
        self.add_edges(neighbors.map(|w| (v, w)));
    }

    pub fn set_neighbors(&mut self, v: usize, neighbors: impl Iterator<Item = usize>) {
        self.adj_lists[v].clear();
        self.adj_lists[v].extend(neighbors);
    }

    pub fn add(&mut self, t: T) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(t);
        self.adj_lists.push(HashSet::new());
        idx
    }

    pub fn get(&self, v: usize) -> Option<&T> {
        self.nodes.get(v)
    }

    pub fn add_edge(&mut self, v: usize, w: usize) {
        if !self.is_in_bounds(v, w) {
            return;
        }

        self.connect_directed(v, w);
        self.connect_directed(w, v);
    }

    pub fn add_directed_edge(&mut self, v: usize, w: usize) {
        self.connect_directed(v, w);
    }

    pub fn remove_edge(&mut self, v: usize, w: usize) {
        if !self.is_in_bounds(v, w) {
            return;
        }

        self.disconnect_directed(v, w);
        self.disconnect_directed(w, v);
    }

    pub fn neighborhood(&self, v: usize) -> impl Iterator<Item = usize> {
        if let Some(set) = self.adj_lists.get(v) {
            return set.iter().copied();
        }

        self.empty.iter().copied()
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

impl<T: Clone> Clone for AdjListGraph<T> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            adj_lists: self.adj_lists.clone(),
            empty: self.empty.clone(),
        }
    }
}

impl<T> Default for AdjListGraph<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::default(),
            adj_lists: Vec::default(),
            empty: HashSet::default(),
        }
    }
}

impl<T> FromIterator<T> for AdjListGraph<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let nodes = iter.into_iter().collect::<Vec<T>>();
        let count = nodes.len();
        Self {
            nodes,
            adj_lists: vec![HashSet::default(); count],
            empty: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashSet, hash::Hash};

    pub fn unordered_eq<T, I1, I2>(a: I1, b: I2) -> bool
    where
        T: Eq + Hash,
        I1: IntoIterator<Item = T>,
        I2: IntoIterator<Item = T>,
    {
        let a: HashSet<_> = a.into_iter().collect();
        let b: HashSet<_> = b.into_iter().collect();

        a == b
    }

    #[test]
    fn test_add() {
        let mut graph = AdjListGraph::new();
        let ten = graph.add(10);
        let two = graph.add(2);
        assert_eq!(graph.size(), 2);
        graph.add_edge(ten, two);
        assert!(graph.is_connected(two, ten));
    }

    #[test]
    fn test_is_connected() {
        let mut graph = AdjListGraph::from_iter(0..2);
        assert!(!graph.is_connected(0, 1));
        graph.add_edge(0, 1);
        assert!(graph.is_connected(0, 1));
    }

    #[test]
    fn test_neighborhood() {
        let mut graph = AdjListGraph::from_iter(0..10);
        for i in 1..6 {
            graph.add_edge(0, i);
        }
        assert!(unordered_eq(graph.neighborhood(0), 1..6));
    }

    #[test]
    fn test_clear_edges() {
        let mut graph = AdjListGraph::from_iter(0..10);
        for i in 1..6 {
            graph.add_edge(0, i);
        }
        for i in 2..6 {
            graph.add_edge(1, i);
        }
        assert!(unordered_eq(graph.neighborhood(0), 1..6));
        assert!(unordered_eq(graph.neighborhood(1), vec![0, 2, 3, 4, 5]));

        graph.clear_edges(1);
        assert!(unordered_eq(graph.neighborhood(0), 2..6));
        assert!(unordered_eq(graph.neighborhood(1), vec![]));
    }
}
