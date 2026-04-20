use ndarray::{Array1, Array2, array, s};
use std::ops::Deref;
use std::{borrow::Cow, marker::PhantomData, path::Path};

use hdf5::{Dataset, DatasetBuilderEmpty, Extents, File, Group, H5Type, Result};
use sprs::{CsMat, CsMatBase};

pub type CsrMat<V> = CsMatBase<V, usize, Vec<usize>, Vec<usize>, Vec<V>, usize>;

#[derive(Clone)]
pub struct H5File {
    file: File,
}

impl H5File {
    pub fn open<P>(path: &P) -> Result<Self>
    where
        P: AsRef<Path> + ?Sized,
    {
        let file = File::open(path)?;
        Ok(Self { file })
    }

    pub fn create<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let file = File::create(path)?;
        Ok(Self { file })
    }

    pub fn read_csr<V: H5Type + Clone + Default + std::fmt::Debug>(
        &self,
        group_name: &str,
    ) -> Result<CsrMat<V>> {
        let group = self.file.group(group_name)?;

        let shape: Vec<usize> = group.attr("shape")?.read_1d()?.to_vec();
        let (rows, cols) = (shape[0], shape[1]);

        let indptr: Vec<usize> = group.dataset("indptr")?.read_1d()?.to_vec();
        let indices: Vec<usize> = group.dataset("indices")?.read_1d()?.to_vec();
        let data: Vec<V> = group.dataset("data")?.read_1d()?.to_vec();

        let csr = CsMat::new_from_unsorted((rows, cols), indptr, indices, data).unwrap();

        Ok(csr)
    }
}

impl Deref for H5File {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

#[derive(Clone)]
pub struct BufferedDataset<'f, T, D> {
    file: Cow<'f, H5File>,
    dataset: Dataset,
    _phantom: PhantomData<(T, D)>,
}

impl<'f, T, D> BufferedDataset<'f, T, D> {
    pub fn open<P>(path: &P, dataset: &str) -> Result<Self>
    where
        P: AsRef<Path> + ?Sized,
    {
        let file = H5File::open(path)?;
        let dataset = file.dataset(dataset)?;
        Ok(BufferedDataset {
            file: Cow::Owned(file),
            dataset,
            _phantom: PhantomData,
        })
    }

    pub fn create<P, S>(path: P, shape: S, dataset: &str) -> Result<Self>
    where
        P: AsRef<Path>,
        S: Into<Extents>,
    {
        let file = H5File::create(path)?;
        let dataset = file.new_dataset::<u64>().shape(shape).create(dataset)?;
        Ok(BufferedDataset {
            file: Cow::Owned(file),
            dataset,
            _phantom: PhantomData,
        })
    }

    pub fn with_file<S>(file: &'f H5File, shape: S, dataset: &str) -> Result<Self>
    where
        S: Into<Extents>,
    {
        let dataset = file.new_dataset::<u64>().shape(shape).create(dataset)?;
        Ok(BufferedDataset {
            file: Cow::Borrowed(file),
            dataset,
            _phantom: PhantomData,
        })
    }

    pub fn add_attr<'n, V, N>(&self, name: N, value: &V) -> Result<()>
    where
        V: H5Type,
        N: Into<&'n str>,
    {
        self.file.new_attr::<V>().create(name)?.write_scalar(value)
    }

    pub fn size(&self) -> usize {
        *self.dataset.shape().first().expect("dataset has no shape")
    }
}

impl<'f, T, D> BufferedDataset<'f, T, D>
where
    T: From<Array1<D>>,
    D: H5Type + Clone,
{
    pub fn write_row(&self, data: T, row: usize) -> Result<()>
    where
        T: Into<Array1<D>>,
    {
        let arr: Array1<D> = data.into();
        self.dataset.write_slice(arr.view(), s![row, ..])
    }
}

impl<'f, T, D> IntoIterator for BufferedDataset<'f, T, D>
where
    T: From<Array1<D>>,
    D: H5Type + Clone,
{
    type Item = T;

    type IntoIter = BufferedDatasetIter<T, D>;

    fn into_iter(self) -> Self::IntoIter {
        BufferedDatasetIter {
            dataset: self.dataset.clone(),
            buffer: ArrayIter::empty(),
            cur: 0,
            len: self.size(),
            _phantom: PhantomData,
        }
    }
}

pub struct BufferedDatasetIter<T, D> {
    dataset: Dataset,
    buffer: ArrayIter<D>,
    cur: usize,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T, D> Iterator for BufferedDatasetIter<T, D>
where
    T: From<Array1<D>>,
    D: H5Type + Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        const BUFFER_SIZE: usize = 50_000;

        if self.cur == self.len {
            return None;
        }

        if let Some(arr) = self.buffer.next() {
            self.cur += 1;
            return Some(arr.into());
        }

        let to = self.len.min(self.cur + BUFFER_SIZE);

        let array = self
            .dataset
            .read_slice_2d(s![self.cur..to, ..])
            .expect("could not read expected rows");

        self.buffer = ArrayIter {
            array,
            cur: 0,
            len: to - self.cur,
        };

        self.cur += 1;
        self.buffer.next().map(Into::into)
    }
}

struct ArrayIter<D> {
    array: Array2<D>,
    cur: usize,
    len: usize,
}

impl<D> ArrayIter<D> {
    fn empty() -> Self {
        Self {
            array: array![[], []],
            cur: 0,
            len: 0,
        }
    }
}

impl<D: H5Type + Clone> Iterator for ArrayIter<D> {
    type Item = Array1<D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.len {
            return None;
        }

        self.cur += 1;
        Some(self.array.row(self.cur - 1).to_owned())
    }
}
