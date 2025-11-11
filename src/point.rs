use ordered_float::OrderedFloat;
use std::cmp::Ordering;

pub trait Point {
    fn distance(&self, other: &Self) -> f32;
}

impl<P> Point for &P
where
    P: Point,
{
    fn distance(&self, other: &Self) -> f32 {
        (*self).distance(*other)
    }
}

#[derive(Debug)]
pub struct Distance<'a, P: Point> {
    pub distance: f32,
    pub key: usize,
    pub point: &'a P,
}

impl<'a, P: Point> Clone for Distance<'a, P> {
    fn clone(&self) -> Self {
        Self {
            distance: self.distance,
            key: self.key,
            point: self.point,
        }
    }
}

impl<'a, P: Point> Distance<'a, P> {
    pub const fn new(distance: f32, key: usize, point: &'a P) -> Self {
        Self {
            distance,
            key,
            point,
        }
    }
}

impl<'a, P: Point> PartialEq for Distance<'a, P> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<'a, P: Point> PartialOrd for Distance<'a, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, P: Point> Eq for Distance<'a, P> {}

impl<'a, P: Point> Ord for Distance<'a, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        match OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance)) {
            Ordering::Equal => self.key.cmp(&other.key),
            ordering => ordering,
        }
    }
}
