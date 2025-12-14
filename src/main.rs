use crate::dataset::BufferedDataset;
use crate::point::{Distance, Point};
use crate::roargraph::{MinK, RoarGraphBuilder, RoarGraphOptions};
use bincode::{deserialize_from, serialize_into};
use ndarray::Array1;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::{collections::HashSet, path::Path};
use serde::{Deserialize, Serialize};
use tracing::info;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

mod adj_list_graph;
mod dataset;
mod point;
mod roargraph;

fn main() {
    tracing_subscriber::registry().with(fmt::layer()).init();

    //let path = Path::new("C:/Users/jonas/Downloads/llama-128-ip.hdf5");
    let path = Path::new("C:/Users/jonas/Downloads/imagenet-align-640-normalized.hdf5");
    let num_corpus = 250_000;
    let corpus = BufferedDataset::<'_, Row<f32>, _>::open(path, "train")
        .unwrap()
        .into_iter()
        .take(num_corpus)
        .collect::<Vec<_>>();
    let num_queries = corpus.len() / 10;
    let queries = BufferedDataset::<'_, Row<f32>, _>::open(path, "learn")
        .unwrap()
        .into_iter()
        .take(num_queries)
        .collect::<Vec<_>>();
    //let knns = BufferedDataset::<'_, Row<usize>, _>::open(path, "neighbors").unwrap();
    //let distances = BufferedDataset::<'_, Row<f32>, _>::open(path, "distances").unwrap();

    let build_ratio = 2;
    let build_count = queries.len() / build_ratio;

    info!("Number of queries: {}", queries.len());
    info!("Number of build queries: {}", build_count);
    info!("Number of eval queries: {}", queries.len() - build_count);
    info!("Corpus size: {}", corpus.len());

    // Ground truth computation
    let ground_truth_file_name = format!(
        "ground_truth-{}.bin",
        path.file_name().unwrap().to_str().unwrap()
    );
    let ground_truth_file = Path::new(ground_truth_file_name.as_str());
    let ground_truth = if ground_truth_file.exists() {
        info!("Reading ground truth...");
        let reader = BufReader::new(File::open(&ground_truth_file).unwrap());
        deserialize_from(reader).unwrap()
    } else {
        info!("Computing ground truth nearest neighbors...");
        let ground_truth = queries
            .par_iter()
            .map(|q| {
                let mut closest = corpus
                    .iter()
                    .enumerate()
                    .map(|(k, d)| Distance::new(d.distance(&q), k, d))
                    .min_k(100);
                closest.sort();
                closest
                    .iter()
                    .map(|d| (d.key, d.distance))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let writer = BufWriter::new(File::create(ground_truth_file_name).unwrap());

        serialize_into(writer, &ground_truth).unwrap();
        ground_truth
    };
    //let ground_truth = knns
    //    .into_iter()
    //    .take(num_queries)
    //    .zip(distances.into_iter().take(num_queries))
    //    .map(|(knn, distance)| {
    //        knn.data
    //            .iter()
    //            .copied()
    //            .zip(distance.data.iter().copied().map(OrderedFloat))
    //            .collect::<Vec<_>>()
    //    })
    //    .collect::<Vec<_>>();

    let ground_truth_keys = ground_truth
        .iter()
        .take(queries.len())
        .map(|v| v.iter().map(|(k, _)| *k).collect())
        .collect::<Vec<_>>();


    let graph_file_name = format!(
        "graph-{}.bin",
        path.file_name().unwrap().to_str().unwrap()
    );
    let graph_file = Path::new(graph_file_name.as_str());
    let graph = if graph_file.exists() {
        info!("Reading graph...");
        let reader = BufReader::new(File::open(&graph_file_name).unwrap());
        deserialize_from(reader).unwrap()
    } else {
        let graph = RoarGraphBuilder::new(RoarGraphOptions { m: 100, l: 500 }).build(
            queries.iter().take(build_count).cloned().collect(),
            corpus,
            ground_truth_keys
                .iter()
                .take(build_count)
                .cloned()
                .collect(),
        );
        let writer = BufWriter::new(File::create(graph_file_name).unwrap());

        serialize_into(writer, &graph).unwrap();
        graph
    };

    let eval_queries = queries
        .iter()
        .skip(build_count)
        .cloned()
        .collect::<Vec<_>>();
    let eval_ground_truth_keys = ground_truth_keys
        .iter()
        .skip(build_count)
        .cloned()
        .collect::<Vec<_>>();

    info!("Evaluating recall...");
    let mut recall: f32 = eval_ground_truth_keys
        .par_iter()
        .zip(eval_queries.par_iter())
        .map(|(knn, query)| {
            let knn = knn.iter().copied().collect::<HashSet<_>>();

            let found = graph.search(query, knn.len());
            let found = found.iter().map(|d| d.key).collect::<HashSet<_>>();

            knn.intersection(&found).count() as f32 / knn.len() as f32
        })
        .sum();
    recall /= eval_ground_truth_keys.len() as f32;
    info!("Recall: {:.4}", recall);
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Row<T> {
    data: Vec<T>,
}

impl<T: Clone> From<Array1<T>> for Row<T> {
    fn from(value: Array1<T>) -> Self {
        Self {
            data: value.to_vec(),
        }
    }
}

impl Point for Row<f32> {
    fn distance(&self, other: &Self) -> f32 {
        // Inner product distance
        -self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f32>()
    }
}
