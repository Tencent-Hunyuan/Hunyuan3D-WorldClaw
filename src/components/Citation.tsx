import { ArrowUpRight, Check, Copy, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { bibtex, paperUrl } from "../data/content";
import { PaperIcon } from "./PaperIcon";
import { Reveal } from "./Reveal";
import { SectionHeader } from "./SectionHeader";

type CopyState = "idle" | "copied" | "error";

export function Citation() {
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const resetTimer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(resetTimer.current), []);

  const scheduleReset = () => {
    window.clearTimeout(resetTimer.current);
    resetTimer.current = window.setTimeout(() => setCopyState("idle"), 2400);
  };

  const copyCitation = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(bibtex);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = bibtex;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.select();
        const didCopy = document.execCommand("copy");
        textArea.remove();
        if (!didCopy) throw new Error("Copy command failed");
      }
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
    scheduleReset();
  };

  return (
    <section id="citation" className="citation section-pad">
      <div className="page-shell">
        <SectionHeader
          index="05"
          eyebrow="Citation"
          title="Cite WorldClaw."
          lede={
            <p>
              If this work supports your research, the BibTeX entry below is the
              canonical reference.
            </p>
          }
        />

        <div className="citation-layout">
          <Reveal className="citation-copy" delay={0.08}>
            <p>
              The paper covers the full agent architecture, the intermediate
              representations shared between stages, and the render-guided
              refinement loop in detail.
            </p>
            <div className="citation-links">
              <a
                className="button button-primary"
                href={paperUrl}
                target="_blank"
                rel="noreferrer"
              >
                <PaperIcon />
                ArXiv
                <span className="button-affordance" aria-hidden="true">
                  <ArrowUpRight weight="bold" />
                </span>
              </a>
            </div>
            <dl className="citation-meta">
              <div>
                <dt>Year</dt>
                <dd>2026</dd>
              </div>
              <div>
                <dt>arXiv</dt>
                <dd>2608.05248</dd>
              </div>
              <div>
                <dt>Entry type</dt>
                <dd>misc</dd>
              </div>
            </dl>
          </Reveal>

          <Reveal className="bibtex-panel" delay={0.14}>
            <div className="bibtex-toolbar">
              <span>worldclaw.bib</span>
              <button
                type="button"
                onClick={copyCitation}
                data-state={copyState}
              >
                {copyState === "copied" ? (
                  <Check weight="bold" />
                ) : copyState === "error" ? (
                  <WarningCircle weight="bold" />
                ) : (
                  <Copy weight="light" />
                )}
                {copyState === "copied"
                  ? "Copied"
                  : copyState === "error"
                    ? "Copy failed"
                    : "Copy"}
              </button>
            </div>
            <pre>
              <code>{bibtex}</code>
            </pre>
            <p className="bibtex-status" role="status">
              {copyState === "copied"
                ? "BibTeX entry copied to the clipboard."
                : copyState === "error"
                  ? "Clipboard access was blocked. Select the text above to copy it."
                  : ""}
            </p>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
