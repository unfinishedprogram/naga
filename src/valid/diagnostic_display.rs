use codespan_reporting::diagnostic::Diagnostic;

pub mod expression_error;

pub trait DiagnosticProvider<T, C> {
    fn get_diagnostic(&self, error: T, context: C) -> Diagnostic<()>;
}
