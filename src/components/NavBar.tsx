import {
  AnimatePresence,
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
  useSpring,
} from "motion/react";
import { useEffect, useState } from "react";
import { paperUrl, sections } from "../data/content";
import { BrandMark } from "./BrandMark";
import { PaperIcon } from "./PaperIcon";
import { ThemeToggle } from "./ThemeToggle";

function useActiveSection() {
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const targets = sections
      .map((section) => document.getElementById(section.id))
      .filter((node): node is HTMLElement => node !== null);

    if (targets.length === 0) return;

    const visible = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          visible.set(entry.target.id, entry.intersectionRatio);
        }
        let best = "";
        let bestRatio = 0;
        for (const [id, ratio] of visible) {
          if (ratio > bestRatio) {
            best = id;
            bestRatio = ratio;
          }
        }
        setActive(bestRatio > 0.05 ? best : "");
      },
      { threshold: [0, 0.05, 0.25, 0.5, 0.75, 1], rootMargin: "-20% 0px -40%" },
    );

    for (const target of targets) observer.observe(target);
    return () => observer.disconnect();
  }, []);

  return active;
}

export function NavBar() {
  const [open, setOpen] = useState(false);
  const [condensed, setCondensed] = useState(false);
  const active = useActiveSection();
  const reduceMotion = useReducedMotion();
  const { scrollYProgress, scrollY } = useScroll();
  const progress = useSpring(scrollYProgress, {
    stiffness: 220,
    damping: 40,
    restDelta: 0.001,
  });

  useMotionValueEvent(scrollY, "change", (value) => {
    setCondensed(value > 40);
  });

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const closeMenu = () => setOpen(false);

  return (
    <>
      <header
        className={`site-nav ${condensed ? "is-condensed" : ""} ${
          open ? "is-menu-open" : ""
        }`}
      >
        <a className="brand" href="#top" aria-label="WorldClaw home">
          <span className="brand-mark" aria-hidden="true">
            <BrandMark />
          </span>
          <span className="brand-word">WorldClaw</span>
        </a>

        <nav className="nav-links" aria-label="Sections">
          {sections.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              aria-current={active === section.id ? "true" : undefined}
              className={active === section.id ? "is-active" : ""}
            >
              <span className="nav-link-index">{section.index}</span>
              {section.label}
            </a>
          ))}
        </nav>

        <div className="nav-tools">
          <a
            className="nav-paper-link"
            href={paperUrl}
            target="_blank"
            rel="noreferrer"
          >
            <PaperIcon />
            <span>arXiv</span>
          </a>
          <ThemeToggle />
          <button
            className="nav-toggle"
            type="button"
            onClick={() => setOpen((value) => !value)}
            aria-expanded={open}
            aria-controls="mobile-menu"
            aria-label={open ? "Close navigation" : "Open navigation"}
          >
            <span className="nav-toggle-bars" aria-hidden="true">
              <span />
              <span />
            </span>
          </button>
        </div>

        <motion.span
          className="nav-progress"
          style={{ scaleX: progress }}
          aria-hidden="true"
        />
      </header>

      <AnimatePresence>
        {open && (
          <motion.div
            id="mobile-menu"
            className="mobile-menu"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.32, ease: [0.32, 0.72, 0, 1] }}
          >
            <nav aria-label="Sections">
              {sections.map((section, index) => (
                <motion.a
                  key={section.id}
                  href={`#${section.id}`}
                  onClick={closeMenu}
                  initial={reduceMotion ? false : { opacity: 0, y: 28 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 12 }}
                  transition={{
                    duration: reduceMotion ? 0 : 0.55,
                    delay: reduceMotion ? 0 : 0.06 + index * 0.05,
                    ease: [0.32, 0.72, 0, 1],
                  }}
                >
                  <span>{section.index}</span>
                  {section.label}
                </motion.a>
              ))}
            </nav>
            <motion.a
              className="mobile-menu-paper"
              href={paperUrl}
              target="_blank"
              rel="noreferrer"
              onClick={closeMenu}
              initial={reduceMotion ? false : { opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: reduceMotion ? 0 : 0.5,
                delay: reduceMotion ? 0 : 0.34,
                ease: [0.32, 0.72, 0, 1],
              }}
            >
              <PaperIcon />
              arXiv
            </motion.a>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
