﻿use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, Argument, ErrorPosition, Handle, TensorLayout};
use digit_layout::DigitLayout;
use std::{cmp::Ordering, iter::zip};

pub struct Args<H: Handle> {
    pub dst_layout: TensorLayout,
    pub dst_base: MutPtr<H>,
    pub src_layout: TensorLayout,
    pub src_base: ConstPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.dst_layout.dt();
        if self.src_layout.dt() != dt {
            return Err(locate_error!());
        }
        let ndim = self.dst_layout.ndim();
        if ndim < 2 || self.src_layout.ndim() != ndim {
            return Err(locate_error!());
        }
        for (&dst, &src) in zip(self.dst_layout.shape(), self.src_layout.shape()) {
            if Argument::merge(&[dst, src]).is_err() {
                return Err(locate_error!());
            }
        }
        Ok(Meta { dt })
    }
}

pub(super) struct Scheme(Vec<isize>);

impl Scheme {
    pub fn new<H: Handle>(args: &Args<H>) -> Result<Self, ErrorPosition> {
        let Args {
            dst_layout: dst_,
            src_layout: src_,
            ..
        } = args;
        // # 检查基本属性
        let unit = dst_.dt().nbytes();
        if src_.dt().nbytes() != unit {
            return Err(locate_error!());
        }
        let ndim = dst_.ndim();
        if src_.ndim() != ndim {
            return Err(locate_error!());
        }
        // # 输入形状
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct Dim {
            len: usize,
            dst: isize,
            src: isize,
        }
        let mut dims = Vec::with_capacity(ndim);
        {
            let dd = dst_.shape();
            let ds = src_.shape();
            let sd = dst_.strides();
            let ss = src_.strides();
            for i in 0..ndim {
                // 合并形状
                let d = *Argument::merge(&[dd[i], ds[i]]).map_err(|_| locate_error!())?;
                // 静态化
                let dim = Dim {
                    len: *d.get_static().ok_or_else(|| locate_error!())?,
                    dst: *sd[i].get_static().ok_or_else(|| locate_error!())?,
                    src: *ss[i].get_static().ok_or_else(|| locate_error!())?,
                };
                // 剔除初始的 1 长维度
                if dim.len != 1 {
                    if dim.dst == 0 {
                        return Err(locate_error!("Reducing is not allowed for reform."));
                    }
                    dims.push(dim);
                }
            }
        }
        // # 排序
        impl PartialOrd for Dim {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Dim {
            /// dst 降序 -> src 降序 -> len 升序
            fn cmp(&self, other: &Self) -> Ordering {
                use Ordering::Equal as Eq;
                match self.dst.cmp(&other.dst) {
                    Eq => match self.src.cmp(&other.src) {
                        Eq => self.len.cmp(&other.len),
                        neq => neq.reverse(),
                    },
                    neq => neq.reverse(),
                }
            }
        }
        dims.sort_unstable();
        // # 合并连续维度
        let mut unit = unit as _;
        let mut ndim = dims.len();
        // ## 合并末尾连续维度到 unit
        for dim in dims.iter_mut().rev() {
            if dim.dst == unit && dim.src == unit {
                unit *= dim.len as isize;
                ndim -= 1;
            } else {
                break;
            }
        }
        dims.truncate(ndim);
        // ## 合并任意连续维度
        for i in (1..dims.len()).rev() {
            let (head, tail) = dims.split_at_mut(i);
            let f = &mut head[i - 1]; // f for front
            let b = &mut tail[0]; // b for back
            let len = b.len as isize;
            if b.dst * len == f.dst && b.src * len == f.src {
                *f = Dim {
                    len: b.len * f.len,
                    dst: b.dst,
                    src: b.src,
                };
                *b = Dim {
                    len: 1,
                    dst: 0,
                    src: 0,
                };
                ndim -= 1;
            }
        }
        // # 合并空间
        let mut layout = vec![0isize; 1 + ndim * 3];
        layout[0] = unit as _;
        for (i, Dim { len, dst, src }) in dims.into_iter().filter(|d| d.len != 1).enumerate() {
            layout[1 + i] = len as _;
            layout[1 + ndim + i] = dst;
            layout[1 + ndim * 2 + i] = src;
        }
        Ok(Self(layout))
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        (self.0.len() - 1) / 3
    }

    #[inline]
    pub fn unit(&self) -> usize {
        self.0[0] as _
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        let ndim = self.ndim();
        unsafe { std::mem::transmute(&self.0[1..][..ndim]) }
    }

    #[inline]
    pub fn dst_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[1 + ndim..][..ndim]
    }

    #[inline]
    pub fn src_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[1 + ndim * 2..][..ndim]
    }
}

#[test]
fn test_scheme() {
    use crate::common_cpu::Handle as Cpu;
    use digit_layout::types::F16;
    use std::ptr::{null, null_mut};

    {
        let shape = [
            Argument::from(4),
            3.into(),
            2.into(),
            1.into(),
            2.into(),
            3.into(),
            4.into(),
        ];
        let args = Args::<Cpu> {
            dst_layout: TensorLayout::new(
                F16,
                &shape,
                &[
                    288.into(), // 4
                    96.into(),  // 3
                    48.into(),  // 2
                    48.into(),  // 1
                    24.into(),  // 2
                    8.into(),   // 3
                    2.into(),   // 4
                ],
            ),
            dst_base: null_mut(),
            src_layout: TensorLayout::new(
                F16,
                &shape,
                &[
                    576.into(), // 4
                    192.into(), // 3
                    96.into(),  // 2
                    48.into(),  // 1
                    8.into(),   // 2
                    16.into(),  // 3
                    2.into(),   // 4
                ],
            ),
            src_base: null(),
        };
        let scheme = Scheme::new(&args).unwrap();
        assert_eq!(scheme.ndim(), 3);
        assert_eq!(scheme.unit(), 8);
        assert_eq!(scheme.shape(), [24, 2, 3]);
        assert_eq!(scheme.dst_strides(), [48, 24, 8]);
        assert_eq!(scheme.src_strides(), [96, 8, 16]);
    }
}
