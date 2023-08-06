use std::ops::Range;

use codespan_reporting::diagnostic::{Diagnostic, Label};

use crate::Span;

pub trait DiagnosticBuilder {
    fn label(self, range: impl Into<Range<usize>>, message: impl Into<String>) -> Self;
}

impl DiagnosticBuilder for Diagnostic<()> {
    fn label(self, range: impl Into<Range<usize>>, message: impl Into<String>) -> Self {
        self.with_labels(vec![Label::primary((), range).with_message(message.into())])
    }
}
