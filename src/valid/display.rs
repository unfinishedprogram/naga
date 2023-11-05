use crate::{ArraySize, Handle, ScalarKind, Type, TypeInner};

impl super::Validator {
    pub fn display_type_handle(&self, handle: Handle<Type>) -> String {
        self.display_type(Some(&self.module.as_ref().unwrap().types[handle].inner))
    }

    pub fn display_type(&self, ty: Option<&TypeInner>) -> String {
        fn print_scalar(kind: &ScalarKind, width: u8) -> String {
            match kind {
                ScalarKind::Sint => format!("i{}", width * 8),
                ScalarKind::Uint => format!("u{}", width * 8),
                ScalarKind::Float => format!("f{}", width * 8),
                ScalarKind::Bool => "bool".into(),
            }
        }

        match ty {
            None => "void".to_owned(),
            Some(inner) => {
                match inner {
                    TypeInner::Scalar { kind, width } => print_scalar(kind, *width),
                    TypeInner::Vector { size, kind, width } => {
                        format!("vec{}<{}>", *size as u8, print_scalar(kind, *width))
                    }
                    TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    } => {
                        format!("mat{}x{}<{}>", *columns as u8, *rows as u8, width * 8)
                    }
                    TypeInner::Atomic { kind, width } => {
                        format!("Atomic<{}>", print_scalar(kind, *width))
                    }
                    TypeInner::Pointer { base, space } => {
                        format!("*{}", self.display_type_handle(*base))
                    }
                    TypeInner::ValuePointer {
                        size,
                        kind,
                        width,
                        space,
                    } => format!("*{}", print_scalar(kind, *width)),
                    TypeInner::Array { base, size, stride } => match *size {
                        ArraySize::Constant(s) => {
                            format!("Array<{}, {s}>", self.display_type_handle(*base))
                        }
                        ArraySize::Dynamic => {
                            format!("Array<{}>", self.display_type_handle(*base))
                        }
                    },

                    // TODO: Improve type printing
                    TypeInner::Image {
                        dim,
                        arrayed,
                        class,
                    } => format!("{dim:?}:{arrayed:?}:{class:?}"),
                    TypeInner::Sampler { comparison } => if *comparison {
                        "sampler_comparison"
                    } else {
                        "sampler"
                    }
                    .into(),
                    TypeInner::AccelerationStructure => "AccelerationStructure".into(),
                    TypeInner::RayQuery => "RayQuery".into(),
                    TypeInner::BindingArray { base, size } => match *size {
                        ArraySize::Constant(s) => {
                            format!("BindingArray<{}, {s}>", self.display_type_handle(*base))
                        }
                        ArraySize::Dynamic => {
                            format!("BindingArray<{}>", self.display_type_handle(*base))
                        }
                    },
                    TypeInner::Struct { members, span } => {
                        let res: String = members
                            .iter()
                            .map(|member| match member.name.as_ref() {
                                Some(name) => {
                                    format!("\t{name}: {},\n", self.display_type_handle(member.ty))
                                }
                                None => self.display_type_handle(member.ty),
                            })
                            .collect();
                        ["{", &res, "}"].join("\n")
                    }
                }
            }
        }
    }
}
