use codespan_reporting::diagnostic::Diagnostic;

use crate::{
    valid::{
        diagnostic_builder::DiagnosticBuilder, diagnostic_display::DiagnosticProvider,
        ExpressionError, FunctionInfo, Validator,
    },
    Expression, Handle, WithSpan,
};

impl
    DiagnosticProvider<WithSpan<ExpressionError>, (&crate::Function, &crate::Module, &FunctionInfo)>
    for Validator
{
    fn get_diagnostic(
        &self,
        error: WithSpan<ExpressionError>,
        (func, module, info): (&crate::Function, &crate::Module, &FunctionInfo),
    ) -> Diagnostic<()> {
        type E = ExpressionError;

        let print_type = |expr: Handle<Expression>| {
            self.display_type(Some(info[expr].ty.inner_with(&module.types)))
        };

        let expr_span = |expr: Handle<Expression>| func.expressions.get_span(expr);

        match error.as_inner() {
            E::DoesntExist => Diagnostic::error().label(0..0, "expression does not exist"),
            E::NotInScope => Diagnostic::error().label(0..0, "expression is not in scope"),
            E::InvalidBaseType(base) => Diagnostic::error().label(
                func.expressions.get_span(*base),
                format!("base type: {} cannot be indexed", (print_type)(*base)),
            ),
            E::InvalidIndexType(index) => Diagnostic::error().label(
                (expr_span)(*index),
                format!("type: {} cannot be used to index", (print_type)(*index)),
            ),
            E::NegativeIndex(index) => {
                Diagnostic::error().label((expr_span)(*index), "index cannot be negative")
            }
            E::IndexOutOfBounds(index, max) => Diagnostic::error().label(
                (expr_span)(*index),
                format!("index is out of bounds: 0..{}", max),
            ),
            // E::IndexMustBeConstant(_) => todo!(),
            // E::FunctionArgumentDoesntExist(_) => todo!(),
            // E::InvalidPointerType(_) => todo!(),
            // E::InvalidArrayType(_) => todo!(),
            // E::InvalidRayQueryType(_) => todo!(),
            // E::InvalidSplatType(_) => todo!(),
            // E::InvalidVectorType(_) => todo!(),
            // E::InvalidSwizzleComponent(_, _) => todo!(),
            // E::Compose(_) => todo!(),
            // E::IndexableLength(_) => todo!(),
            E::InvalidUnaryOperandType(operand, value) => Diagnostic::error().label(
                (expr_span)(*value),
                format!(
                    "operation `{operand:?}` cannot be applied to type: `{}`",
                    (print_type)(*value)
                ),
            ),
            E::InvalidBinaryOperandTypes(operand, left, right) => Diagnostic::error().label(
                error.spans().next().unwrap().0,
                format!(
                    "operation `{operand:?}` cannot be applied to types: `{}` and `{}`",
                    (print_type)(*left),
                    (print_type)(*right)
                ),
            ),
            // E::InvalidSelectTypes => todo!(),
            // E::InvalidBooleanVector(_) => todo!(),
            // E::InvalidFloatArgument(_) => todo!(),
            // E::Type(_) => todo!(),
            // E::ExpectedGlobalVariable => todo!(),
            // E::ExpectedGlobalOrArgument => todo!(),
            // E::ExpectedBindingArrayType(_) => todo!(),
            // E::ExpectedImageType(_) => todo!(),
            // E::ExpectedSamplerType(_) => todo!(),
            // E::InvalidImageClass(_) => todo!(),
            // E::InvalidDerivative => todo!(),
            // E::InvalidImageArrayIndex => todo!(),
            // E::InvalidImageOtherIndex => todo!(),
            // E::InvalidImageArrayIndexType(_) => todo!(),
            // E::InvalidImageOtherIndexType(_) => todo!(),
            // E::InvalidImageCoordinateType(_, _) => todo!(),
            // E::ComparisonSamplingMismatch {
            //     image,
            //     sampler,
            //     has_ref,
            // } => todo!(),
            // E::InvalidSampleOffset(_, _) => todo!(),
            // E::InvalidDepthReference(_) => todo!(),
            // E::InvalidDepthSampleLevel => todo!(),
            // E::InvalidGatherLevel => todo!(),
            // E::InvalidGatherComponent(_) => todo!(),
            // E::InvalidGatherDimension(_) => todo!(),
            // E::InvalidSampleLevelExactType(_) => todo!(),
            // E::InvalidSampleLevelBiasType(_) => todo!(),
            // E::InvalidSampleLevelGradientType(_, _) => todo!(),
            // E::InvalidCastArgument => todo!(),
            // E::WrongArgumentCount(_) => todo!(),
            // E::InvalidArgumentType(_, _, _) => todo!(),
            // E::InvalidAtomicResultType(_) => todo!(),
            // E::InvalidWorkGroupUniformLoadResultType(_) => todo!(),
            // E::MissingCapabilities(_) => todo!(),
            _ => error.diagnostic(),
        }
    }
}
