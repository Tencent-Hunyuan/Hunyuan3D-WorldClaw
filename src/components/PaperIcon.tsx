import { FileText } from "@phosphor-icons/react";

/**
 * A neutral preprint glyph rather than the arXiv monogram: the arXiv mark is
 * illegible at the 14px these links render at, and the label already reads
 * "arXiv". This also stays correct while the link still points at the PDF.
 */
export function PaperIcon() {
  return <FileText weight="light" aria-hidden="true" focusable="false" />;
}
