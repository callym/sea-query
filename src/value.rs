//! Container for all SQL value types.
use std::fmt::Write;

#[cfg(feature = "with-json")]
use serde_json::Value as Json;
#[cfg(feature = "with-json")]
use std::str::from_utf8;

#[cfg(feature = "with-chrono")]
use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime};

#[cfg(feature = "with-rust_decimal")]
use rust_decimal::Decimal;

#[cfg(feature = "with-bigdecimal")]
use bigdecimal::BigDecimal;

#[cfg(feature = "with-uuid")]
use uuid::Uuid;

use crate::ColumnType;

/// Value variants
///
/// We want Value to be exactly 1 pointer sized, so anything larger should be boxed.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(Option<bool>),
    TinyInt(Option<i8>),
    SmallInt(Option<i16>),
    Int(Option<i32>),
    BigInt(Option<i64>),
    TinyUnsigned(Option<u8>),
    SmallUnsigned(Option<u16>),
    Unsigned(Option<u32>),
    BigUnsigned(Option<u64>),
    Float(Option<f32>),
    Double(Option<f64>),
    String(Option<Box<String>>),

    #[allow(clippy::box_vec)]
    Bytes(Option<Box<Vec<u8>>>),

    #[cfg(feature = "with-json")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-json")))]
    Json(Option<Box<Json>>),

    #[cfg(feature = "with-chrono")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-chrono")))]
    Date(Option<Box<NaiveDate>>),

    #[cfg(feature = "with-chrono")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-chrono")))]
    Time(Option<Box<NaiveTime>>),

    #[cfg(feature = "with-chrono")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-chrono")))]
    DateTime(Option<Box<NaiveDateTime>>),

    #[cfg(feature = "with-chrono")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-chrono")))]
    DateTimeWithTimeZone(Option<Box<DateTime<FixedOffset>>>),

    #[cfg(feature = "with-uuid")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-uuid")))]
    Uuid(Option<Box<Uuid>>),

    #[cfg(feature = "with-rust_decimal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-rust_decimal")))]
    Decimal(Option<Box<Decimal>>),

    #[cfg(feature = "with-bigdecimal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "with-bigdecimal")))]
    BigDecimal(Option<Box<BigDecimal>>),
}

pub trait ValueType: Sized {
    fn try_from(v: Value) -> Result<Self, ValueTypeErr>;

    fn unwrap(v: Value) -> Self {
        Self::try_from(v).unwrap()
    }

    fn type_name() -> String;

    fn column_type() -> ColumnType;
}

#[derive(Debug)]
pub struct ValueTypeErr;

impl std::error::Error for ValueTypeErr {}

impl std::fmt::Display for ValueTypeErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Value type mismatch")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Values(pub Vec<Value>);

#[derive(Debug, PartialEq)]
pub enum ValueTuple {
    One(Value),
    Two(Value, Value),
    Three(Value, Value, Value),
    Four(Value, Value, Value, Value),
    Five(Value, Value, Value, Value, Value),
    Six(Value, Value, Value, Value, Value, Value),
    Seven(Value, Value, Value, Value, Value, Value, Value),
    Eight(Value, Value, Value, Value, Value, Value, Value, Value),
    Nine(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Ten(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Eleven(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Twelve(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Thirteen(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Fourteen(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Fifteen(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Sixteen(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Seventeen(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Eighteen(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Nineteen(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
    Twenty(
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
        Value,
    ),
}

pub trait IntoValueTuple {
    fn into_value_tuple(self) -> ValueTuple;
}

pub trait FromValueTuple: Sized {
    fn from_value_tuple<I>(i: I) -> Self
    where
        I: IntoValueTuple;
}

pub trait Nullable {
    fn null() -> Value;
}

impl Value {
    pub fn unwrap<T>(self) -> T
    where
        T: ValueType,
    {
        T::unwrap(self)
    }
}

macro_rules! type_to_value {
    ( $type: ty, $name: ident, $col_type: expr ) => {
        impl From<$type> for Value {
            fn from(x: $type) -> Value {
                Value::$name(Some(x))
            }
        }

        impl Nullable for $type {
            fn null() -> Value {
                Value::$name(None)
            }
        }

        impl ValueType for $type {
            fn try_from(v: Value) -> Result<Self, ValueTypeErr> {
                match v {
                    Value::$name(Some(x)) => Ok(x),
                    _ => Err(ValueTypeErr),
                }
            }

            fn type_name() -> String {
                stringify!($type).to_owned()
            }

            fn column_type() -> ColumnType {
                use ColumnType::*;
                $col_type
            }
        }
    };
}

macro_rules! type_to_box_value {
    ( $type: ty, $name: ident, $col_type: expr ) => {
        impl From<$type> for Value {
            fn from(x: $type) -> Value {
                Value::$name(Some(Box::new(x)))
            }
        }

        impl Nullable for $type {
            fn null() -> Value {
                Value::$name(None)
            }
        }

        impl ValueType for $type {
            fn try_from(v: Value) -> Result<Self, ValueTypeErr> {
                match v {
                    Value::$name(Some(x)) => Ok(*x),
                    _ => Err(ValueTypeErr),
                }
            }

            fn type_name() -> String {
                stringify!($type).to_owned()
            }

            fn column_type() -> ColumnType {
                use ColumnType::*;
                $col_type
            }
        }
    };
}

type_to_value!(bool, Bool, Boolean);
type_to_value!(i8, TinyInt, TinyInteger(None));
type_to_value!(i16, SmallInt, SmallInteger(None));
type_to_value!(i32, Int, Integer(None));
type_to_value!(i64, BigInt, BigInteger(None));

// FIXME: edit this mapping after we added unsigned column types
type_to_value!(u8, TinyUnsigned, TinyInteger(None));
type_to_value!(u16, SmallUnsigned, SmallInteger(None));
type_to_value!(u32, Unsigned, Integer(None));
type_to_value!(u64, BigUnsigned, BigInteger(None));
type_to_value!(f32, Float, Float(None));
type_to_value!(f64, Double, Double(None));

impl<'a> From<&'a [u8]> for Value {
    fn from(x: &'a [u8]) -> Value {
        Value::Bytes(Some(Box::<Vec<u8>>::new(x.into())))
    }
}

impl<'a> From<&'a str> for Value {
    fn from(x: &'a str) -> Value {
        let string: String = x.into();
        Value::String(Some(Box::new(string)))
    }
}

impl<'a> Nullable for &'a str {
    fn null() -> Value {
        Value::String(None)
    }
}

impl<T> From<Option<T>> for Value
where
    T: Into<Value> + Nullable,
{
    fn from(x: Option<T>) -> Value {
        match x {
            Some(v) => v.into(),
            None => T::null(),
        }
    }
}

impl<T> ValueType for Option<T>
where
    T: ValueType + Nullable,
{
    fn try_from(v: Value) -> Result<Self, ValueTypeErr> {
        if v == T::null() {
            Ok(None)
        } else {
            Ok(Some(T::try_from(v)?))
        }
    }

    fn type_name() -> String {
        format!("Option<{}>", T::type_name())
    }

    fn column_type() -> ColumnType {
        T::column_type()
    }
}

type_to_box_value!(Vec<u8>, Bytes, Binary(None));
type_to_box_value!(String, String, String(None));

#[cfg(feature = "with-json")]
#[cfg_attr(docsrs, doc(cfg(feature = "with-json")))]
mod with_json {
    use super::*;

    type_to_box_value!(Json, Json, Json);
}

#[cfg(feature = "with-chrono")]
#[cfg_attr(docsrs, doc(cfg(feature = "with-chrono")))]
mod with_chrono {
    use super::*;
    use chrono::{Offset, TimeZone};

    type_to_box_value!(NaiveDate, Date, Date);
    type_to_box_value!(NaiveTime, Time, Time(None));
    type_to_box_value!(NaiveDateTime, DateTime, DateTime(None));

    impl<Tz> From<DateTime<Tz>> for Value
    where
        Tz: TimeZone,
    {
        fn from(x: DateTime<Tz>) -> Value {
            let v = DateTime::<FixedOffset>::from_utc(x.naive_utc(), x.offset().fix());
            Value::DateTimeWithTimeZone(Some(Box::new(v)))
        }
    }

    impl Nullable for DateTime<FixedOffset> {
        fn null() -> Value {
            Value::DateTimeWithTimeZone(None)
        }
    }

    impl ValueType for DateTime<FixedOffset> {
        fn try_from(v: Value) -> Result<Self, ValueTypeErr> {
            match v {
                Value::DateTimeWithTimeZone(Some(x)) => Ok(*x),
                _ => Err(ValueTypeErr),
            }
        }

        fn type_name() -> String {
            stringify!(DateTime<FixedOffset>).to_owned()
        }

        fn column_type() -> ColumnType {
            ColumnType::TimestampWithTimeZone(None)
        }
    }
}

#[cfg(feature = "with-rust_decimal")]
#[cfg_attr(docsrs, doc(cfg(feature = "with-rust_decimal")))]
mod with_rust_decimal {
    use super::*;

    type_to_box_value!(Decimal, Decimal, Decimal(None));
}

#[cfg(feature = "with-bigdecimal")]
#[cfg_attr(docsrs, doc(cfg(feature = "with-bigdecimal")))]
mod with_bigdecimal {
    use super::*;

    type_to_box_value!(BigDecimal, BigDecimal, Decimal(None));
}

#[cfg(feature = "with-uuid")]
#[cfg_attr(docsrs, doc(cfg(feature = "with-uuid")))]
mod with_uuid {
    use super::*;

    type_to_box_value!(Uuid, Uuid, Uuid);
}

#[allow(unused_macros)]
macro_rules! box_to_opt_ref {
    ( $v: expr ) => {
        match $v {
            Some(v) => Some(v.as_ref()),
            None => None,
        }
    };
}

impl Value {
    pub fn is_json(&self) -> bool {
        #[cfg(feature = "with-json")]
        return matches!(self, Self::Json(_));
        #[cfg(not(feature = "with-json"))]
        return false;
    }
    #[cfg(feature = "with-json")]
    pub fn as_ref_json(&self) -> Option<&Json> {
        match self {
            Self::Json(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::Json"),
        }
    }
    #[cfg(not(feature = "with-json"))]
    pub fn as_ref_json(&self) -> Option<&bool> {
        panic!("not Value::Json")
    }
}

impl Value {
    pub fn is_date(&self) -> bool {
        #[cfg(feature = "with-chrono")]
        return matches!(self, Self::Date(_));
        #[cfg(not(feature = "with-chrono"))]
        return false;
    }
    #[cfg(feature = "with-chrono")]
    pub fn as_ref_date(&self) -> Option<&NaiveDate> {
        match self {
            Self::Date(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::Date"),
        }
    }
    #[cfg(not(feature = "with-chrono"))]
    pub fn as_ref_date(&self) -> Option<&bool> {
        panic!("not Value::Date")
    }
}

impl Value {
    pub fn is_time(&self) -> bool {
        #[cfg(feature = "with-chrono")]
        return matches!(self, Self::Time(_));
        #[cfg(not(feature = "with-chrono"))]
        return false;
    }
    #[cfg(feature = "with-chrono")]
    pub fn as_ref_time(&self) -> Option<&NaiveTime> {
        match self {
            Self::Time(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::Time"),
        }
    }
    #[cfg(not(feature = "with-chrono"))]
    pub fn as_ref_time(&self) -> Option<&bool> {
        panic!("not Value::Time")
    }
}

impl Value {
    pub fn is_date_time(&self) -> bool {
        #[cfg(feature = "with-chrono")]
        return matches!(self, Self::DateTime(_));
        #[cfg(not(feature = "with-chrono"))]
        return false;
    }
    #[cfg(feature = "with-chrono")]
    pub fn as_ref_date_time(&self) -> Option<&NaiveDateTime> {
        match self {
            Self::DateTime(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::DateTime"),
        }
    }
    #[cfg(not(feature = "with-chrono"))]
    pub fn as_ref_date_time(&self) -> Option<&bool> {
        panic!("not Value::DateTime")
    }
}

impl Value {
    pub fn is_date_time_with_time_zone(&self) -> bool {
        #[cfg(feature = "with-chrono")]
        return matches!(self, Self::DateTimeWithTimeZone(_));
        #[cfg(not(feature = "with-chrono"))]
        return false;
    }
    #[cfg(feature = "with-chrono")]
    pub fn as_ref_date_time_with_time_zone(&self) -> Option<&DateTime<FixedOffset>> {
        match self {
            Self::DateTimeWithTimeZone(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::DateTimeWithTimeZone"),
        }
    }
    #[cfg(not(feature = "with-chrono"))]
    pub fn as_ref_date_time_with_time_zone(&self) -> Option<&bool> {
        panic!("not Value::DateTimeWithTimeZone")
    }
}

impl Value {
    pub fn is_decimal(&self) -> bool {
        #[cfg(feature = "with-rust_decimal")]
        return matches!(self, Self::Decimal(_));
        #[cfg(not(feature = "with-rust_decimal"))]
        return false;
    }
    #[cfg(feature = "with-rust_decimal")]
    pub fn as_ref_decimal(&self) -> Option<&Decimal> {
        match self {
            Self::Decimal(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::Decimal"),
        }
    }
    #[cfg(feature = "with-rust_decimal")]
    pub fn decimal_to_f64(&self) -> Option<f64> {
        use rust_decimal::prelude::ToPrimitive;
        self.as_ref_decimal().map(|d| d.to_f64().unwrap())
    }
    #[cfg(not(feature = "with-rust_decimal"))]
    pub fn as_ref_decimal(&self) -> Option<&bool> {
        panic!("not Value::Decimal")
    }
    #[cfg(not(feature = "with-rust_decimal"))]
    pub fn decimal_to_f64(&self) -> Option<f64> {
        None
    }
}

impl Value {
    pub fn is_big_decimal(&self) -> bool {
        #[cfg(feature = "with-bigdecimal")]
        return matches!(self, Self::BigDecimal(_));
        #[cfg(not(feature = "with-bigdecimal"))]
        return false;
    }
    #[cfg(feature = "with-bigdecimal")]
    pub fn as_ref_big_decimal(&self) -> Option<&BigDecimal> {
        match self {
            Self::BigDecimal(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::BigDecimal"),
        }
    }
    #[cfg(feature = "with-bigdecimal")]
    pub fn big_decimal_to_f64(&self) -> Option<f64> {
        use bigdecimal::ToPrimitive;
        self.as_ref_big_decimal().map(|d| d.to_f64().unwrap())
    }
    #[cfg(not(feature = "with-bigdecimal"))]
    pub fn as_ref_big_decimal(&self) -> Option<&bool> {
        panic!("not Value::BigDecimal")
    }
    #[cfg(not(feature = "with-bigdecimal"))]
    pub fn big_decimal_to_f64(&self) -> Option<f64> {
        None
    }
}

impl Value {
    pub fn is_uuid(&self) -> bool {
        #[cfg(feature = "with-uuid")]
        return matches!(self, Self::Uuid(_));
        #[cfg(not(feature = "with-uuid"))]
        return false;
    }
    #[cfg(feature = "with-uuid")]
    pub fn as_ref_uuid(&self) -> Option<&Uuid> {
        match self {
            Self::Uuid(v) => box_to_opt_ref!(v),
            _ => panic!("not Value::Uuid"),
        }
    }
    #[cfg(not(feature = "with-uuid"))]
    pub fn as_ref_uuid(&self) -> Option<&bool> {
        panic!("not Value::Uuid")
    }
}

impl IntoIterator for ValueTuple {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            ValueTuple::One(a) => vec![a].into_iter(),
            ValueTuple::Two(a, b) => vec![a, b].into_iter(),
            ValueTuple::Three(a, b, c) => vec![a, b, c].into_iter(),
            ValueTuple::Four(a, b, c, d) => vec![a, b, c, d].into_iter(),
            ValueTuple::Five(a, b, c, d, e) => vec![a, b, c, d, e].into_iter(),
            ValueTuple::Six(a, b, c, d, e, f) => vec![a, b, c, d, e, f].into_iter(),
            ValueTuple::Seven(a, b, c, d, e, f, g) => vec![a, b, c, d, e, f, g].into_iter(),
            ValueTuple::Eight(a, b, c, d, e, f, g, h) => vec![a, b, c, d, e, f, g, h].into_iter(),
            ValueTuple::Nine(a, b, c, d, e, f, g, h, i) => {
                vec![a, b, c, d, e, f, g, h, i].into_iter()
            }
            ValueTuple::Ten(a, b, c, d, e, f, g, h, i, j) => {
                vec![a, b, c, d, e, f, g, h, i, j].into_iter()
            }
            ValueTuple::Eleven(a, b, c, d, e, f, g, h, i, j, k) => {
                vec![a, b, c, d, e, f, g, h, i, j, k].into_iter()
            }
            ValueTuple::Twelve(a, b, c, d, e, f, g, h, i, j, k, l) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l].into_iter()
            }
            ValueTuple::Thirteen(a, b, c, d, e, f, g, h, i, j, k, l, m) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m].into_iter()
            }
            ValueTuple::Fourteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n].into_iter()
            }
            ValueTuple::Fifteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].into_iter()
            }
            ValueTuple::Sixteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p].into_iter()
            }
            ValueTuple::Seventeen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q].into_iter()
            }
            ValueTuple::Eighteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r].into_iter()
            }
            ValueTuple::Nineteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s].into_iter()
            }
            ValueTuple::Twenty(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) => {
                vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t].into_iter()
            }
        }
    }
}

impl IntoValueTuple for ValueTuple {
    fn into_value_tuple(self) -> ValueTuple {
        self
    }
}

impl<A> IntoValueTuple for A
where
    A: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::One(self.into())
    }
}

impl<A, B> IntoValueTuple for (A, B)
where
    A: Into<Value>,
    B: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Two(self.0.into(), self.1.into())
    }
}
impl<A, B, C> IntoValueTuple for (A, B, C)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Three(self.0.into(), self.1.into(), self.2.into())
    }
}
impl<A, B, C, D> IntoValueTuple for (A, B, C, D)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Four(self.0.into(), self.1.into(), self.2.into(), self.3.into())
    }
}
impl<A, B, C, D, E> IntoValueTuple for (A, B, C, D, E)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Five(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
        )
    }
}
impl<A, B, C, D, E, F> IntoValueTuple for (A, B, C, D, E, F)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Six(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
        )
    }
}
impl<A, B, C, D, E, F, G> IntoValueTuple for (A, B, C, D, E, F, G)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Seven(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H> IntoValueTuple for (A, B, C, D, E, F, G, H)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Eight(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I> IntoValueTuple for (A, B, C, D, E, F, G, H, I)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Nine(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J> IntoValueTuple for (A, B, C, D, E, F, G, H, I, J)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Ten(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K> IntoValueTuple for (A, B, C, D, E, F, G, H, I, J, K)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Eleven(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L> IntoValueTuple for (A, B, C, D, E, F, G, H, I, J, K, L)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Twelve(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Thirteen(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
    N: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Fourteen(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
            self.13.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
    N: Into<Value>,
    O: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Fifteen(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
            self.13.into(),
            self.14.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
    N: Into<Value>,
    O: Into<Value>,
    P: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Sixteen(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
            self.13.into(),
            self.14.into(),
            self.15.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
    N: Into<Value>,
    O: Into<Value>,
    P: Into<Value>,
    Q: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Seventeen(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
            self.13.into(),
            self.14.into(),
            self.15.into(),
            self.16.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
    N: Into<Value>,
    O: Into<Value>,
    P: Into<Value>,
    Q: Into<Value>,
    R: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Eighteen(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
            self.13.into(),
            self.14.into(),
            self.15.into(),
            self.16.into(),
            self.17.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
    N: Into<Value>,
    O: Into<Value>,
    P: Into<Value>,
    Q: Into<Value>,
    R: Into<Value>,
    S: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Nineteen(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
            self.13.into(),
            self.14.into(),
            self.15.into(),
            self.16.into(),
            self.17.into(),
            self.18.into(),
        )
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T> IntoValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T)
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
    D: Into<Value>,
    E: Into<Value>,
    F: Into<Value>,
    G: Into<Value>,
    H: Into<Value>,
    I: Into<Value>,
    J: Into<Value>,
    K: Into<Value>,
    L: Into<Value>,
    M: Into<Value>,
    N: Into<Value>,
    O: Into<Value>,
    P: Into<Value>,
    Q: Into<Value>,
    R: Into<Value>,
    S: Into<Value>,
    T: Into<Value>,
{
    fn into_value_tuple(self) -> ValueTuple {
        ValueTuple::Twenty(
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
            self.8.into(),
            self.9.into(),
            self.10.into(),
            self.11.into(),
            self.12.into(),
            self.13.into(),
            self.14.into(),
            self.15.into(),
            self.16.into(),
            self.17.into(),
            self.18.into(),
            self.19.into(),
        )
    }
}

impl<A> FromValueTuple for A
where
    A: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::One(u) => u.unwrap(),
            _ => panic!("not ValueTuple::One"),
        }
    }
}

impl<A, B> FromValueTuple for (A, B)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Two(a, b) => (a.unwrap(), b.unwrap()),
            _ => panic!("not ValueTuple::Two"),
        }
    }
}
impl<A, B, C> FromValueTuple for (A, B, C)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Three(a, b, c) => (a.unwrap(), b.unwrap(), c.unwrap()),
            _ => panic!("not ValueTuple::Three"),
        }
    }
}
impl<A, B, C, D> FromValueTuple for (A, B, C, D)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Four(a, b, c, d) => (a.unwrap(), b.unwrap(), c.unwrap(), d.unwrap()),
            _ => panic!("not ValueTuple::Four"),
        }
    }
}
impl<A, B, C, D, E> FromValueTuple for (A, B, C, D, E)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Five(a, b, c, d, e) => {
                (a.unwrap(), b.unwrap(), c.unwrap(), d.unwrap(), e.unwrap())
            }
            _ => panic!("not ValueTuple::Five"),
        }
    }
}
impl<A, B, C, D, E, F> FromValueTuple for (A, B, C, D, E, F)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Six(a, b, c, d, e, f) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
            ),
            _ => panic!("not ValueTuple::Six"),
        }
    }
}
impl<A, B, C, D, E, F, G> FromValueTuple for (A, B, C, D, E, F, G)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Seven(a, b, c, d, e, f, g) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
            ),
            _ => panic!("not ValueTuple::Seven"),
        }
    }
}
impl<A, B, C, D, E, F, G, H> FromValueTuple for (A, B, C, D, E, F, G, H)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Eight(a, b, c, d, e, f, g, h) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
            ),
            _ => panic!("not ValueTuple::Eight"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I> FromValueTuple for (A, B, C, D, E, F, G, H, I)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Nine(a, b, c, d, e, f, g, h, i) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
            ),
            _ => panic!("not ValueTuple::Nine"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J> FromValueTuple for (A, B, C, D, E, F, G, H, I, J)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Ten(a, b, c, d, e, f, g, h, i, j) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
            ),
            _ => panic!("not ValueTuple::Ten"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K> FromValueTuple for (A, B, C, D, E, F, G, H, I, J, K)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Eleven(a, b, c, d, e, f, g, h, i, j, k) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
            ),
            _ => panic!("not ValueTuple::Eleven"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L> FromValueTuple for (A, B, C, D, E, F, G, H, I, J, K, L)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Twelve(a, b, c, d, e, f, g, h, i, j, k, l) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
            ),
            _ => panic!("not ValueTuple::Twelve"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Thirteen(a, b, c, d, e, f, g, h, i, j, k, l, m) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
            ),
            _ => panic!("not ValueTuple::Thirteen"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
    N: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Fourteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
                n.unwrap(),
            ),
            _ => panic!("not ValueTuple::Fourteen"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
    N: Into<Value> + ValueType,
    O: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Fifteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
                n.unwrap(),
                o.unwrap(),
            ),
            _ => panic!("not ValueTuple::Fifteen"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
    N: Into<Value> + ValueType,
    O: Into<Value> + ValueType,
    P: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Sixteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
                n.unwrap(),
                o.unwrap(),
                p.unwrap(),
            ),
            _ => panic!("not ValueTuple::Sixteen"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
    N: Into<Value> + ValueType,
    O: Into<Value> + ValueType,
    P: Into<Value> + ValueType,
    Q: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Seventeen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
                n.unwrap(),
                o.unwrap(),
                p.unwrap(),
                q.unwrap(),
            ),
            _ => panic!("not ValueTuple::Seventeen"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
    N: Into<Value> + ValueType,
    O: Into<Value> + ValueType,
    P: Into<Value> + ValueType,
    Q: Into<Value> + ValueType,
    R: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Eighteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
                n.unwrap(),
                o.unwrap(),
                p.unwrap(),
                q.unwrap(),
                r.unwrap(),
            ),
            _ => panic!("not ValueTuple::Eighteen"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
    N: Into<Value> + ValueType,
    O: Into<Value> + ValueType,
    P: Into<Value> + ValueType,
    Q: Into<Value> + ValueType,
    R: Into<Value> + ValueType,
    S: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Nineteen(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
                n.unwrap(),
                o.unwrap(),
                p.unwrap(),
                q.unwrap(),
                r.unwrap(),
                s.unwrap(),
            ),
            _ => panic!("not ValueTuple::Nineteen"),
        }
    }
}
impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T> FromValueTuple
    for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T)
where
    A: Into<Value> + ValueType,
    B: Into<Value> + ValueType,
    C: Into<Value> + ValueType,
    D: Into<Value> + ValueType,
    E: Into<Value> + ValueType,
    F: Into<Value> + ValueType,
    G: Into<Value> + ValueType,
    H: Into<Value> + ValueType,
    I: Into<Value> + ValueType,
    J: Into<Value> + ValueType,
    K: Into<Value> + ValueType,
    L: Into<Value> + ValueType,
    M: Into<Value> + ValueType,
    N: Into<Value> + ValueType,
    O: Into<Value> + ValueType,
    P: Into<Value> + ValueType,
    Q: Into<Value> + ValueType,
    R: Into<Value> + ValueType,
    S: Into<Value> + ValueType,
    T: Into<Value> + ValueType,
{
    fn from_value_tuple<Z>(i: Z) -> Self
    where
        Z: IntoValueTuple,
    {
        match i.into_value_tuple() {
            ValueTuple::Twenty(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) => (
                a.unwrap(),
                b.unwrap(),
                c.unwrap(),
                d.unwrap(),
                e.unwrap(),
                f.unwrap(),
                g.unwrap(),
                h.unwrap(),
                i.unwrap(),
                j.unwrap(),
                k.unwrap(),
                l.unwrap(),
                m.unwrap(),
                n.unwrap(),
                o.unwrap(),
                p.unwrap(),
                q.unwrap(),
                r.unwrap(),
                s.unwrap(),
                t.unwrap(),
            ),
            _ => panic!("not ValueTuple::Twenty"),
        }
    }
}

/// Escape a SQL string literal
pub fn escape_string(string: &str) -> String {
    string
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("'", "\\'")
        .replace("\0", "\\0")
        .replace("\x08", "\\b")
        .replace("\x09", "\\t")
        .replace("\x1a", "\\z")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
}

/// Unescape a SQL string literal
pub fn unescape_string(input: &str) -> String {
    let mut escape = false;
    let mut output = String::new();
    for c in input.chars() {
        if !escape && c == '\\' {
            escape = true;
        } else if escape {
            write!(
                output,
                "{}",
                match c {
                    '0' => '\0',
                    'b' => '\x08',
                    't' => '\x09',
                    'z' => '\x1a',
                    'n' => '\n',
                    'r' => '\r',
                    c => c,
                }
            )
            .unwrap();
            escape = false;
        } else {
            write!(output, "{}", c).unwrap();
        }
    }
    output
}

/// Convert value to json value
#[allow(clippy::many_single_char_names)]
#[cfg(feature = "with-json")]
#[cfg_attr(docsrs, doc(cfg(feature = "with-json")))]
pub fn sea_value_to_json_value(value: &Value) -> Json {
    use crate::{CommonSqlQueryBuilder, QueryBuilder};

    match value {
        Value::Bool(None)
        | Value::TinyInt(None)
        | Value::SmallInt(None)
        | Value::Int(None)
        | Value::BigInt(None)
        | Value::TinyUnsigned(None)
        | Value::SmallUnsigned(None)
        | Value::Unsigned(None)
        | Value::BigUnsigned(None)
        | Value::Float(None)
        | Value::Double(None)
        | Value::String(None)
        | Value::Bytes(None)
        | Value::Json(None) => Json::Null,
        #[cfg(feature = "with-rust_decimal")]
        Value::Decimal(None) => Json::Null,
        #[cfg(feature = "with-bigdecimal")]
        Value::BigDecimal(None) => Json::Null,
        #[cfg(feature = "with-uuid")]
        Value::Uuid(None) => Json::Null,
        Value::Bool(Some(b)) => Json::Bool(*b),
        Value::TinyInt(Some(v)) => (*v).into(),
        Value::SmallInt(Some(v)) => (*v).into(),
        Value::Int(Some(v)) => (*v).into(),
        Value::BigInt(Some(v)) => (*v).into(),
        Value::TinyUnsigned(Some(v)) => (*v).into(),
        Value::SmallUnsigned(Some(v)) => (*v).into(),
        Value::Unsigned(Some(v)) => (*v).into(),
        Value::BigUnsigned(Some(v)) => (*v).into(),
        Value::Float(Some(v)) => (*v).into(),
        Value::Double(Some(v)) => (*v).into(),
        Value::String(Some(s)) => Json::String(s.as_ref().clone()),
        Value::Bytes(Some(s)) => Json::String(from_utf8(s).unwrap().to_string()),
        Value::Json(Some(v)) => v.as_ref().clone(),
        #[cfg(feature = "with-chrono")]
        Value::Date(_) => CommonSqlQueryBuilder.value_to_string(value).into(),
        #[cfg(feature = "with-chrono")]
        Value::Time(_) => CommonSqlQueryBuilder.value_to_string(value).into(),
        #[cfg(feature = "with-chrono")]
        Value::DateTime(_) => CommonSqlQueryBuilder.value_to_string(value).into(),
        #[cfg(feature = "with-chrono")]
        Value::DateTimeWithTimeZone(_) => CommonSqlQueryBuilder.value_to_string(value).into(),
        #[cfg(feature = "with-rust_decimal")]
        Value::Decimal(Some(v)) => {
            use rust_decimal::prelude::ToPrimitive;
            v.as_ref().to_f64().unwrap().into()
        }
        #[cfg(feature = "with-bigdecimal")]
        Value::BigDecimal(Some(v)) => {
            use bigdecimal::ToPrimitive;
            v.as_ref().to_f64().unwrap().into()
        }
        #[cfg(feature = "with-uuid")]
        Value::Uuid(Some(v)) => Json::String(v.to_string()),
    }
}

impl Values {
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_1() {
        let test = r#" "abc" "#;
        assert_eq!(escape_string(test), r#" \"abc\" "#.to_owned());
        assert_eq!(unescape_string(escape_string(test).as_str()), test);
    }

    #[test]
    fn test_escape_2() {
        let test = "a\nb\tc";
        assert_eq!(escape_string(test), "a\\nb\\tc".to_owned());
        assert_eq!(unescape_string(escape_string(test).as_str()), test);
    }

    #[test]
    fn test_escape_3() {
        let test = "a\\b";
        assert_eq!(escape_string(test), "a\\\\b".to_owned());
        assert_eq!(unescape_string(escape_string(test).as_str()), test);
    }

    #[test]
    fn test_escape_4() {
        let test = "a\"b";
        assert_eq!(escape_string(test), "a\\\"b".to_owned());
        assert_eq!(unescape_string(escape_string(test).as_str()), test);
    }

    #[test]
    fn test_value() {
        macro_rules! test_value {
            ( $type: ty, $val: literal ) => {
                let val: $type = $val;
                let v: Value = val.into();
                let out: $type = v.unwrap();
                assert_eq!(out, val);
            };
        }

        test_value!(u8, 255);
        test_value!(u16, 65535);
        test_value!(i8, 127);
        test_value!(i16, 32767);
        test_value!(i32, 1073741824);
        test_value!(i64, 8589934592);
    }

    #[test]
    fn test_option_value() {
        macro_rules! test_some_value {
            ( $type: ty, $val: literal ) => {
                let val: Option<$type> = Some($val);
                let v: Value = val.into();
                let out: $type = v.unwrap();
                assert_eq!(out, val.unwrap());
            };
        }

        macro_rules! test_none {
            ( $type: ty, $name: ident ) => {
                let val: Option<$type> = None;
                let v: Value = val.into();
                assert_eq!(v, Value::$name(None));
            };
        }

        test_some_value!(u8, 255);
        test_some_value!(u16, 65535);
        test_some_value!(i8, 127);
        test_some_value!(i16, 32767);
        test_some_value!(i32, 1073741824);
        test_some_value!(i64, 8589934592);

        test_none!(u8, TinyUnsigned);
        test_none!(u16, SmallUnsigned);
        test_none!(i8, TinyInt);
        test_none!(i16, SmallInt);
        test_none!(i32, Int);
        test_none!(i64, BigInt);
    }

    #[test]
    fn test_box_value() {
        let val: String = "hello".to_owned();
        let v: Value = val.clone().into();
        let out: String = v.unwrap();
        assert_eq!(out, val);
    }

    #[test]
    fn test_value_tuple() {
        assert_eq!(
            1i32.into_value_tuple(),
            ValueTuple::One(Value::Int(Some(1)))
        );
        assert_eq!(
            "b".into_value_tuple(),
            ValueTuple::One(Value::String(Some(Box::new("b".to_owned()))))
        );
        assert_eq!(
            (1i32, "b").into_value_tuple(),
            ValueTuple::Two(
                Value::Int(Some(1)),
                Value::String(Some(Box::new("b".to_owned())))
            )
        );
        assert_eq!(
            (1i32, 2.4f64, "b").into_value_tuple(),
            ValueTuple::Three(
                Value::Int(Some(1)),
                Value::Double(Some(2.4)),
                Value::String(Some(Box::new("b".to_owned())))
            )
        );
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn test_from_value_tuple() {
        let mut val = 1i32;
        let original = val.clone();
        val = FromValueTuple::from_value_tuple(val);
        assert_eq!(val, original);

        let mut val = "b".to_owned();
        let original = val.clone();
        val = FromValueTuple::from_value_tuple(val);
        assert_eq!(val, original);

        let mut val = (1i32, "b".to_owned());
        let original = val.clone();
        val = FromValueTuple::from_value_tuple(val);
        assert_eq!(val, original);

        let mut val = (1i32, 2.4f64, "b".to_owned());
        let original = val.clone();
        val = FromValueTuple::from_value_tuple(val);
        assert_eq!(val, original);
    }

    #[test]
    fn test_value_tuple_iter() {
        let mut iter = (1i32).into_value_tuple().into_iter();
        assert_eq!(iter.next().unwrap(), Value::Int(Some(1)));
        assert_eq!(iter.next(), None);

        let mut iter = (1i32, 2.4f64).into_value_tuple().into_iter();
        assert_eq!(iter.next().unwrap(), Value::Int(Some(1)));
        assert_eq!(iter.next().unwrap(), Value::Double(Some(2.4)));
        assert_eq!(iter.next(), None);

        let mut iter = (1i32, 2.4f64, "b").into_value_tuple().into_iter();
        assert_eq!(iter.next().unwrap(), Value::Int(Some(1)));
        assert_eq!(iter.next().unwrap(), Value::Double(Some(2.4)));
        assert_eq!(
            iter.next().unwrap(),
            Value::String(Some(Box::new("b".to_owned())))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[cfg(feature = "with-json")]
    fn test_json_value() {
        let json = serde_json::json! {{
            "a": 25.0,
            "b": "hello",
        }};
        let value: Value = json.clone().into();
        let out: Json = value.unwrap();
        assert_eq!(out, json);
    }

    #[test]
    #[cfg(feature = "with-chrono")]
    fn test_chrono_value() {
        let timestamp = chrono::NaiveDate::from_ymd(2020, 1, 1).and_hms(2, 2, 2);
        let value: Value = timestamp.into();
        let out: NaiveDateTime = value.unwrap();
        assert_eq!(out, timestamp);
    }

    #[test]
    #[cfg(feature = "with-chrono")]
    fn test_chrono_timezone_value() {
        let timestamp = DateTime::parse_from_rfc3339("2020-01-01T02:02:02+08:00").unwrap();
        let value: Value = timestamp.into();
        let out: DateTime<FixedOffset> = value.unwrap();
        assert_eq!(out, timestamp);
    }

    #[test]
    #[cfg(feature = "with-chrono")]
    fn test_chrono_query() {
        use crate::*;

        let string = "2020-01-01T02:02:02+08:00";
        let timestamp = DateTime::parse_from_rfc3339(string).unwrap();

        let query = Query::select().expr(Expr::val(timestamp)).to_owned();

        let formatted = "2020-01-01 02:02:02 +08:00";

        assert_eq!(
            query.to_string(MysqlQueryBuilder),
            format!("SELECT '{}'", formatted)
        );
        assert_eq!(
            query.to_string(PostgresQueryBuilder),
            format!("SELECT '{}'", formatted)
        );
        assert_eq!(
            query.to_string(SqliteQueryBuilder),
            format!("SELECT '{}'", formatted)
        );
    }

    #[test]
    #[cfg(feature = "with-uuid")]
    fn test_uuid_value() {
        let uuid = uuid::Uuid::parse_str("936DA01F9ABD4d9d80C702AF85C822A8").unwrap();
        let value: Value = uuid.into();
        let out: uuid::Uuid = value.unwrap();
        assert_eq!(out, uuid);
    }

    #[test]
    #[cfg(feature = "with-rust_decimal")]
    fn test_decimal_value() {
        use std::str::FromStr;

        let num = "2.02";
        let val = Decimal::from_str(num).unwrap();
        let v: Value = val.into();
        let out: Decimal = v.unwrap();
        assert_eq!(out.to_string(), num);
    }
}
