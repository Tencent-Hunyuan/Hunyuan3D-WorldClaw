import katex from "katex";
import "katex/dist/katex.min.css";

interface MathFormulaProps {
  formula: string;
  className?: string;
  displayMode?: boolean;
}

export function MathFormula({
  formula,
  className,
  displayMode = false,
}: MathFormulaProps) {
  const markup = katex.renderToString(formula, {
    displayMode,
    throwOnError: false,
    strict: false,
  });

  return (
    <span
      className={`math-formula ${className ?? ""}`.trim()}
      aria-label={formula}
      dangerouslySetInnerHTML={{ __html: markup }}
    />
  );
}
