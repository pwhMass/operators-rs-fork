﻿use super::{args::Meta, fill_pos, Args, Rope, Seq, SinCosTable};
use crate::{
    get_static,
    opencl::{ClDevice, KernelCache},
    shape_not_support, strides_not_support, type_not_support,
    utils::sizeof,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use clrt::bindings::cl_int;
use clrt::Invalid;
use digit_layout::{
    // types::{F16, U32},
    types::{F32, U32},
};
use std::{alloc::Layout, ffi::CString, iter::zip};

pub struct Operator(KernelCache);

const MAX_THREADS_PER_BLOCK: usize = 512;

impl Rope<ClDevice> for Operator {
    fn build_sincos<QA>(
        _dt: digit_layout::DigitLayout,
        _nctx: usize,
        _dh: usize,
        _queue_alloc: &QA,
    ) -> SinCosTable<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        SinCosTable {
            nctx: 0,
            mem: _queue_alloc.alloc(0),
        }
    }

    fn build_pos<I, QA>(
        _dt: digit_layout::DigitLayout,
        _nt: usize,
        _iter: I,
        _queue_alloc: &QA,
    ) -> QA::DevMem
    where
        I: IntoIterator<Item = Seq>,
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut blob = _queue_alloc.alloc(Layout::array::<u32>(_nt).unwrap().size());
        let mut host = vec![0u32; _nt];
        fill_pos(&mut host, _iter);
        let queue = _queue_alloc.queue();
        let mut map = queue.map_mut(&mut blob, Invalid);
        let ([], mem, []) = (unsafe { map.write_only_slice().align_to_mut::<f32>() }) else {
            panic!()
        };
        for (dst, src) in zip(mem, &host) {
            *dst = *src as _;
        }
        queue.unmap(map);
        blob
    }
}

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(_node: &Self::TopoNode) -> Self {
        let options = CString::new("").unwrap();
        let program = _node
            .context()
            .build_from_source(include_str!("rope.cl"), options);
        Self(KernelCache::new(program))
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { dt_t, dt_p, .. } = _args.meta()?;
        // if dt_t == F16 || dt_p == U32 {
        if dt_t == F32 || dt_p == U32 {
            Ok(0)
        } else {
            Err(type_not_support(""))
        }
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta {
            dt_t, dt_p, nt, dh, ..
        } = args.meta()?;

        // if dt_t != F16 || dt_p != U32 {
        if dt_t != F32 || dt_p != U32 {
            return Err(type_not_support("").into());
        }

        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            theta,
            ..
        } = args;
        let &[_, nh, _] = t_layout.shape() else {
            unreachable!()
        };
        let &[st, sh, sd] = t_layout.strides() else {
            unreachable!()
        };
        let &[sp] = p_layout.strides() else {
            unreachable!()
        };

        get_static! {
            nt nh dh
            st sh sd
            sp
        }
        let unit = sizeof(dt_t)? as isize;
        if sd != unit || sp != size_of::<u32>() as isize {
            return Err(strides_not_support("").into());
        };

        let dh = dh / 2;
        let st = (st / unit / 2) as i32;
        let sh = (sh / unit / 2) as i32;

        if MAX_THREADS_PER_BLOCK % dh != 0 {
            return Err(shape_not_support("").into());
        }

        let max_nh_l = (MAX_THREADS_PER_BLOCK / dh).min(nh);
        let nh_l = (1..=max_nh_l).rev().find(|nhl| nh % nhl == 0).unwrap();
        let nh_h = nh / nh_l;

        let global_workoffset = [0, 0];
        let global_worksize = [(nt * nh_l) as usize, (nh_h * dh) as usize];
        let local_worksize = [nh_l as usize, dh as usize];

        // let name = "rope_f16";
        let name = "rope_f32";
        let mut kernel = self.0.get_kernel(name).unwrap();

        kernel
            .set_arg(0, t_base)
            .set_arg(1, st as cl_int)
            .set_arg(2, sh as cl_int)
            .set_arg(3, p_base)
            .set_arg(4, theta)
            .launch(
                &global_workoffset,
                &global_worksize,
                &local_worksize,
                _queue_alloc.queue(),
                None,
            );

        self.0.set_kernel(name, kernel);

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::Args;
    use crate::{Hardware, TensorLayout};
    use digit_layout::{
        // types::{F16, F64, U32},
        types::{F32, F64, U32},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt_t: DigitLayout, dt_p: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            t_layout: TensorLayout::new_dyn(dt_t, &[dyn_(); 3], &[dyn_(); 3]),
            t_base: null_mut(),
            p_layout: TensorLayout::new_dyn(dt_p, &[dyn_()], &[dyn_()]),
            p_base: null(),
            sin_layout: TensorLayout::new_dyn(dt_t, &[dyn_(); 2], &[dyn_(); 2]),
            sin_base: null(),
            cos_layout: TensorLayout::new_dyn(dt_t, &[dyn_(); 2], &[dyn_(); 2]),
            cos_base: null(),
            theta: 0.,
        }
    }

    fn args<H: Hardware>(
        dt_t: DigitLayout,
        dt_p: DigitLayout,
        nt: usize,
        nh: usize,
        dh: usize,
        theta: f32,
        t_base: *mut H::Byte,
        p_base: *const H::Byte,
    ) -> Args<H> {
        use std::ptr::null;
        Args {
            t_layout: TensorLayout::new_contiguous(dt_t, &[nt, nh, dh]),
            t_base,
            p_layout: TensorLayout::new_contiguous(dt_p, &[nt]),
            p_base,
            sin_layout: TensorLayout::new_contiguous(dt_t, &[0, dh]),
            sin_base: null(),
            cos_layout: TensorLayout::new_contiguous(dt_t, &[0, dh]),
            cos_base: null(),
            theta,
        }
    }

    #[test]
    fn test_compute() {
        use super::{super::common_cpu::Operator as RefOp, Operator};
        use crate::{
            common_cpu::{Cpu, ThisThread},
            opencl::ClDevice,
            test_utils::{Diff, ErrorCollector},
            Operator as _,
        };
        use clrt::{Invalid, Platform};
        use digit_layout::types as ty;
        use rand::Rng;
        // use half::f16;
        // use rayon::iter::IntoParallelIterator;
        use std::{iter::zip, time::Instant};

        let mut cpu_op = RefOp::new(&Cpu);
        for platform in Platform::all() {
            for device in platform.devices() {
                println!("device: {}", device.name());

                let context = device.context();
                let queue = context.queue();
                let mut cl_op = Operator::new(&ClDevice::new(context.clone()));
                cpu_op.scheme(&dyn_args(F64, U32), 0).unwrap();
                // cl_op.scheme(&dyn_args(F16, U32), 0).unwrap();
                cl_op.scheme(&dyn_args(F32, U32), 0).unwrap();

                const NT: usize = 7;
                let nh = 32;
                let dh = 64;

                let mut t = vec![0.0f64; NT * nh * dh];
                rand::thread_rng().fill(&mut t[..]);
                let p: [u32; NT] = [0, 1, 2, 3, 7, 8, 1];
                // let mut t_svm = context.malloc::<f16>(NT * nh * dh);
                let mut t_svm = context.malloc::<f32>(NT * nh * dh);
                let mut p_svm = context.malloc::<u32>(7);

                let mut map = queue.map_mut(&mut t_svm, Invalid);
                // let ([], mem, []) = (unsafe { map.write_only_slice().align_to_mut::<f16>() })
                let ([], mem, []) = (unsafe { map.write_only_slice().align_to_mut::<f32>() })
                else {
                    panic!()
                };
                for (dst, src) in zip(mem, &t) {
                    *dst = *src as _;
                    // *dst =f16::from_f64(*src);
                }
                queue.unmap(map);

                let mut map = queue.map_mut(&mut p_svm, Invalid);
                let ([], mem, []) = (unsafe { map.write_only_slice().align_to_mut::<u32>() })
                else {
                    panic!()
                };
                for (dst, src) in zip(mem, &p) {
                    *dst = *src as _;
                }
                queue.unmap(map);

                let time = Instant::now();
                cl_op
                    .launch(
                        &args(
                            // ty::F16,
                            ty::F32,
                            ty::U32,
                            NT,
                            nh,
                            dh,
                            1e4,
                            t_svm.as_mut_ptr().cast(),
                            p_svm.as_ptr().cast(),
                        ),
                        &mut [],
                        &queue,
                    )
                    .unwrap();
                queue.finish();
                let cl_time = time.elapsed();

                //CPU
                let mut t_ref = t;
                let time = Instant::now();
                cpu_op
                    .launch(
                        &args(
                            F64,
                            U32,
                            NT,
                            nh,
                            dh,
                            1e4,
                            t_ref.as_mut_ptr().cast(),
                            p.as_ptr().cast(),
                        ),
                        &mut [],
                        &ThisThread,
                    )
                    .unwrap();
                let cpu_time = time.elapsed();

                let map = queue.map(&mut t_svm);
                // let ([], y_ans, []) = (unsafe { map.align_to::<f16>() }) else {
                let ([], y_ans, []) = (unsafe { map.align_to::<f32>() }) else {
                    panic!()
                };
                // for (index, i) in y_ans.iter().enumerate() {
                //     print!("{}: {} ", index + 1, i);
                // }
                // println!();
                // println!();
                // for (index, i) in t_ref.iter().enumerate() {
                //     print!("{}: {} ", index + 1, i);
                // }
                // println!();
                // println!();

                let diff = t_ref
                    .into_iter()
                    .zip(y_ans)
                    // .map(|(a, b)| Diff::new(a, b.to_f64()))
                    .map(|(a, b)| Diff::new(a, *b as _))
                    .collect::<Vec<_>>();
                queue.unmap(map);

                // let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 1e-3);
                let mut ec = ErrorCollector::new(f32::EPSILON as f64, 1e-3);
                diff.into_iter().for_each(|diff| ec.push(diff));
                println!("{ec}");
                println!("cl: {cl_time:?} / cpu: {cpu_time:?}");

                let (out, count) = ec.summary();
                assert!(out * 1000 <= count);
                // assert!(2 <= 1);
            }
        }
    }
}
