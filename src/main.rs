#![allow(unused)]

use crate::dataset::BufferedDataset;
use crate::roargraph::{MinK, RoarGraphBuilder, RoarGraphOptions};
use crate::row::Row;
use bincode::{deserialize_from, serialize_into};
use hnsw_itu::{Distance, IndexVis, Point};
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufWriter};
use std::time::Duration;
use std::{collections::HashSet, path::Path};
use tracing::info;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

mod adj_list_graph;
mod dataset;
mod roargraph;
mod row;

fn main() {
    tracing_subscriber::registry().with(fmt::layer()).init();

    let path = Path::new(
        "/Users/jonasuj/Code/master-thesis/datasets/data.exclude/imagenet-align-640-normalized.hdf5",
    );
    //let path = Path::new("/Users/jonasuj/Code/master-thesis/datasets/data.exclude/yi-128-ip.hdf5");
    //let path = Path::new("/Users/jonasuj/Downloads/llama-128-ip.hdf5");
    //let path = Path::new("/Users/jonasuj/Downloads/imagenet-align-640-normalized.hdf5");
    let outdir = "data.exclude";
    let num_corpus = 256921;
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
        "{outdir}/ground_truth-{}.bin",
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
        "{outdir}/graph-{}.bin",
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
            corpus.iter().cloned().collect(),
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

    let mut vis_count = vec![0usize; corpus.len()];
    let ef = 10;
    info!("Evaluating recall...");
    let mut result: Vec<(f32, Duration)> = eval_ground_truth_keys
        .iter()
        .zip(eval_queries.iter())
        .map(|(knn, query)| {
            let knn = knn.iter().copied().collect::<HashSet<_>>();

            let mut vis = HashSet::new();
            let start = std::time::Instant::now();
            let found = graph.search_vis(query, ef, &ef, &mut vis);
            let elapsed = start.elapsed();
            let found = found.iter().map(|d| d.key).collect::<HashSet<_>>();

            for v in vis {
                vis_count[v.key] += 1;
            }

            (knn.intersection(&found).count() as f32 / ef as f32, elapsed)
        })
        .collect();
    let recall = result.iter().map(|(r, _)| r).sum::<f32>() / eval_ground_truth_keys.len() as f32;
    let spq = result.iter().map(|(_, e)| e).sum::<Duration>() / eval_ground_truth_keys.len() as u32;
    info!("Recall@{ef}: {:.4}", recall);
    info!("SPQ: {:?}", spq);

    let mut frequency_map = vec![0usize; corpus.len()];
    for closest in ground_truth.iter().take(build_count) {
        for &(idx, _) in closest {
            frequency_map[idx] += 1;
        }
    }

    let outfile = format!(
        "frequency-{}.txt",
        path.file_name().unwrap().to_str().unwrap()
    );
    let mut file = File::create(outfile).unwrap();
    for ((p, f), v) in graph
        .graph
        .adj_lists()
        .into_iter()
        .zip(frequency_map.into_iter())
        .zip(vis_count.into_iter())
    {
        writeln!(file, "{},{},{}", p.len(), f, v).unwrap();
    }

    let outfile = format!(
        "roargraph-{}.txt",
        path.file_name().unwrap().to_str().unwrap()
    );
    let mut file = File::create(outfile).unwrap();
    for p in graph.graph.adj_lists().into_iter() {
        writeln!(file, "{}", p.len()).unwrap();
    }
}
