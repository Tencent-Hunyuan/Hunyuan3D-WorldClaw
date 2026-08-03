import type { ReactNode } from "react";
import { Reveal } from "./Reveal";

interface SectionHeaderProps {
  index: string;
  eyebrow: string;
  title?: ReactNode;
  lede?: ReactNode;
}

export function SectionHeader({
  index,
  eyebrow,
  title,
  lede,
}: SectionHeaderProps) {
  return (
    <Reveal className="section-header">
      <div className="section-header-rail">
        <span className="section-index">{index}</span>
        <span className="section-eyebrow">{eyebrow}</span>
        <span className="section-rule" aria-hidden="true" />
      </div>
      {title || lede ? (
        <div className="section-header-body">
          {title ? <h2>{title}</h2> : null}
          {lede ? <div className="section-lede">{lede}</div> : null}
        </div>
      ) : null}
    </Reveal>
  );
}
