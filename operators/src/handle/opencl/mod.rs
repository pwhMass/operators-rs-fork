use crate::{Alloc, Hardware, Pool, QueueAlloc, QueueOf};
use clrt::{CommandQueue, Context, Kernel, Program, SvmBlob, SvmByte};
use std::{collections::HashMap, ffi::CString, sync::RwLock};

#[repr(transparent)]
pub struct ClDevice(Context);

impl Hardware for ClDevice {
    type Byte = SvmByte;
    type Queue<'ctx> = CommandQueue;
}

impl ClDevice {
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(context)
    }

    #[inline]
    pub(crate) fn context(&self) -> &Context {
        &self.0
    }
}

impl Alloc<SvmBlob> for Context {
    #[inline]
    fn alloc(&self, size: usize) -> SvmBlob {
        self.malloc::<usize>(size)
    }

    #[inline]
    fn free(&self, _mem: SvmBlob) {}
}

impl Alloc<SvmBlob> for CommandQueue {
    #[inline]
    fn alloc(&self, size: usize) -> SvmBlob {
        self.ctx().malloc::<usize>(size)
    }

    #[inline]
    fn free(&self, mem: SvmBlob) {
        self.free(mem, None)
    }
}

impl QueueAlloc for CommandQueue {
    type Hardware = ClDevice;
    type DevMem = SvmBlob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}

pub(crate) struct KernelCache {
    program: Program,
    kernels: RwLock<HashMap<String, Pool<Kernel>>>,
}

impl KernelCache {
    pub fn new(program: Program) -> Self {
        Self {
            program,
            kernels: Default::default(),
        }
    }

    pub fn get_kernel(&self, name: &str) -> Option<Kernel> {
        let kernels = self.kernels.read().unwrap();
        if let Some(pool) = kernels.get(name) {
            return pool
                .pop()
                .or_else(|| self.program.get_kernel(&CString::new(name).unwrap()));
        }
        drop(kernels);

        let kernel = self.program.get_kernel(&CString::new(name).unwrap())?;

        let mut kernels = self.kernels.write().unwrap();
        kernels.entry(name.into()).or_insert_with(|| Pool::new());

        Some(kernel)
    }

    pub fn set_kernel(&self, name: &str, kernel: Kernel) {
        self.kernels.read().unwrap().get(name).unwrap().push(kernel)
    }
}
