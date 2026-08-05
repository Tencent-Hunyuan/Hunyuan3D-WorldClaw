import { ArrowUp } from "@phosphor-icons/react";
import { attributionsUrl } from "../data/attributions";
import { paperUrl, sections } from "../data/content";
import { BrandMark } from "./BrandMark";

export function Footer() {
  return (
    <footer className="footer">
      <div className="page-shell footer-inner">
        <div className="footer-brand">
          <a className="brand" href="#top" aria-label="Back to the top">
            <span className="brand-mark" aria-hidden="true">
              <BrandMark />
            </span>
            <span className="brand-word">WorldClaw</span>
          </a>
          <p>
            Agentic open-world 3D scene generation at scale. Built by the
            Tencent Hunyuan3D team.
          </p>
        </div>

        <nav className="footer-nav" aria-label="Footer">
          <span>Sections</span>
          {sections.map((section) => (
            <a key={section.id} href={`#${section.id}`}>
              {section.label}
            </a>
          ))}
        </nav>

        <nav className="footer-nav" aria-label="Resources">
          <span>Resources</span>
          <a href={paperUrl} target="_blank" rel="noreferrer">
            ArXiv
          </a>
          <a href="#citation">BibTeX</a>
          <a href={attributionsUrl}>Attributions</a>
        </nav>
      </div>

      <div className="page-shell footer-notice">
        <p>
          Unless otherwise stated, all images, 3D assets, and other visual
          content displayed on this website are owned by Tencent. No rights or
          licenses to Tencent-owned content are granted. All rights reserved.
        </p>
        <p>
          Certain visual content incorporates third-party materials, which
          remain owned by their respective creators and are licensed under
          Creative Commons Attribution. Attribution information for such
          third-party materials is available <a href={attributionsUrl}>here</a>.
        </p>
      </div>

      <div className="page-shell footer-base">
        <p>© 2026 Tencent Hunyuan3D. All rights reserved.</p>
        <a className="footer-top" href="#top">
          Back to top
          <ArrowUp weight="bold" aria-hidden="true" />
        </a>
      </div>
    </footer>
  );
}
