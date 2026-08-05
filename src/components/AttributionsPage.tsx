import { ArrowLeft, ArrowUpRight } from "@phosphor-icons/react";
import {
  attributions,
  CC_BY_NAME,
  CC_BY_URL,
  homeUrl,
} from "../data/attributions";
import { BrandMark } from "./BrandMark";
import { ThemeToggle } from "./ThemeToggle";

export function AttributionsPage() {
  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <div className="grain" aria-hidden="true" />

      <header className="subpage-bar">
        <div className="page-shell subpage-bar-inner">
          <a className="brand" href={homeUrl}>
            <span className="brand-mark" aria-hidden="true">
              <BrandMark />
            </span>
            <span className="brand-word">WorldClaw</span>
          </a>
          <div className="subpage-bar-actions">
            <a className="subpage-back" href={homeUrl}>
              <ArrowLeft weight="bold" aria-hidden="true" />
              Back to the project
            </a>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main id="main-content" className="subpage section-pad">
        <div className="page-shell">
          <div className="section-header">
            <div className="section-header-rail">
              <span className="section-eyebrow">Attributions</span>
              <span className="section-rule" aria-hidden="true" />
            </div>
            <div className="subpage-intro">
              <h1>Third-party materials.</h1>
              <p>
                Unless otherwise stated, all images, 3D assets, and other visual
                content displayed on this website are owned by Tencent. Certain
                visual content incorporates the third-party materials listed
                below, which remain owned by their respective creators and are
                licensed under{" "}
                <a href={CC_BY_URL} target="_blank" rel="noreferrer">
                  {CC_BY_NAME} 4.0
                </a>
                .
              </p>
            </div>
          </div>

          <ol className="attribution-list">
            {attributions.map((item) => (
              <li key={item.url}>
                <a
                  className="attribution-title"
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {item.title}
                  <ArrowUpRight weight="bold" aria-hidden="true" />
                </a>
                <p className="attribution-meta">
                  by {item.author}, licensed under{" "}
                  <a href={CC_BY_URL} target="_blank" rel="noreferrer">
                    {CC_BY_NAME}
                  </a>
                  .
                </p>
              </li>
            ))}
          </ol>
        </div>
      </main>

      <footer className="footer">
        <div className="page-shell footer-base">
          <p>© 2026 Tencent Hunyuan3D. All rights reserved.</p>
          <a className="footer-top" href={homeUrl}>
            Back to the project
          </a>
        </div>
      </footer>
    </>
  );
}
