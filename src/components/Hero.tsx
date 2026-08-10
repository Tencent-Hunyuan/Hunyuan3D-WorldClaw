import {
  ArrowDown,
  ArrowUpRight,
  Flower,
  Leaf,
  Snowflake,
  Sun,
} from "@phosphor-icons/react";
import {
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
} from "motion/react";
import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  useEffect,
  useState,
} from "react";
import { codeUrl, contributors, paperUrl, seasons } from "../data/content";
import { GithubIcon } from "./GithubIcon";
import { PaperIcon } from "./PaperIcon";

type ThemeStyle = CSSProperties & {
  "--season-accent": string;
  "--season-soft": string;
  "--season-canvas": string;
};

const particles = Array.from({ length: 26 }, (_, index) => ({
  id: index,
  left: (index * 37 + 11) % 100,
  delay: (index * 0.41) % 6,
  duration: 5.8 + (index % 7) * 0.62,
  drift: ((index * 29) % 90) - 45,
  size: 5 + (index % 4) * 2,
}));

const seasonIcons = {
  spring: Flower,
  summer: Sun,
  autumn: Leaf,
  winter: Snowflake,
};

const ROTATE_INTERVAL = 10000;

export function Hero() {
  const [activeIndex, setActiveIndex] = useState(0);
  const reduceMotion = useReducedMotion();
  const activeSeason = seasons[activeIndex];
  const baseUrl = import.meta.env.BASE_URL;
  const tiltX = useMotionValue(0);
  const tiltY = useMotionValue(0);
  const springX = useSpring(tiltX, { stiffness: 170, damping: 26 });
  const springY = useSpring(tiltY, { stiffness: 170, damping: 26 });
  const imageRotateX = useTransform(springX, (value) => `${value}deg`);
  const imageRotateY = useTransform(springY, (value) => `${value}deg`);

  /*
   * activeIndex is a dependency so a manual pick restarts the countdown rather
   * than inheriting the remainder of the current tick — otherwise a click made
   * moments before a tick would flip the season again almost immediately.
   */
  useEffect(() => {
    if (reduceMotion) return;
    const timer = window.setInterval(() => {
      if (document.hidden) return;
      setActiveIndex((index) => (index + 1) % seasons.length);
    }, ROTATE_INTERVAL);
    return () => window.clearInterval(timer);
  }, [reduceMotion, activeIndex]);

  const themeStyle: ThemeStyle = {
    "--season-accent": activeSeason.accent,
    "--season-soft": activeSeason.accentSoft,
    "--season-canvas": activeSeason.canvas,
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (reduceMotion || event.pointerType !== "mouse") return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width - 0.5;
    const y = (event.clientY - bounds.top) / bounds.height - 0.5;
    tiltX.set(y * -3.5);
    tiltY.set(x * 4.5);
  };

  const resetTilt = () => {
    tiltX.set(0);
    tiltY.set(0);
  };

  const handleSeasonKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : seasons.length - 1;
    const next = (activeIndex + offset) % seasons.length;
    setActiveIndex(next);
    const buttons =
      event.currentTarget.querySelectorAll<HTMLButtonElement>("button");
    buttons[next]?.focus();
  };

  return (
    <section
      id="top"
      className="hero"
      style={themeStyle}
      data-season={activeSeason.id}
    >
      <div className="hero-wash" aria-hidden="true">
        {seasons.map((season, index) => (
          <span
            key={season.id}
            className={index === activeIndex ? "is-active" : ""}
            style={
              {
                "--wash-soft": season.accentSoft,
                "--wash-accent": season.accent,
                "--wash-canvas": season.canvas,
              } as CSSProperties
            }
          />
        ))}
      </div>
      <div className="hero-topography" aria-hidden="true" />
      <div
        className={`particle-field particle-${activeSeason.particle}`}
        aria-hidden="true"
      >
        {/* Stable keys: re-keying per season restarts 26 CSS animations mid-switch. */}
        {particles.map((particle) => (
          <span
            key={particle.id}
            style={
              {
                "--particle-left": `${particle.left}%`,
                "--particle-delay": `${particle.delay}s`,
                "--particle-duration": `${particle.duration}s`,
                "--particle-drift": `${particle.drift}px`,
                "--particle-size": `${particle.size}px`,
              } as CSSProperties
            }
          />
        ))}
      </div>

      <div className="hero-layout page-shell">
        <motion.div
          className="hero-copy"
          initial={reduceMotion ? false : { opacity: 0, y: 22 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: reduceMotion ? 0 : 0.9,
            ease: [0.16, 1, 0.3, 1],
          }}
        >
          <p className="hero-kicker">
            <span className="hero-kicker-dot" aria-hidden="true" />
            Tencent Hunyuan3D Research
          </p>
          <div className="hero-title-lockup">
            <h1>WorldClaw</h1>
            <p>Agentic 3D open-world generation at scale</p>
          </div>
          <p className="hero-summary">
            Turn one open-ended prompt into an explicit, explorable, and editable
            3D world. Agentic planning keeps global terrain coherent while
            selectively building rich local detail.
          </p>

          <div className="hero-actions">
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
            <a
              className="button button-quiet"
              href={codeUrl}
              target="_blank"
              rel="noreferrer"
            >
              <GithubIcon />
              GitHub
              <span className="button-affordance" aria-hidden="true">
                <ArrowUpRight weight="bold" />
              </span>
            </a>
            <a className="button button-quiet" href="#results">
              Explore worlds
              <span className="button-affordance" aria-hidden="true">
                <ArrowDown weight="bold" />
              </span>
            </a>
          </div>
        </motion.div>

        <motion.div
          className="hero-visual"
          initial={reduceMotion ? false : { opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            duration: reduceMotion ? 0 : 1.1,
            delay: reduceMotion ? 0 : 0.1,
            ease: [0.16, 1, 0.3, 1],
          }}
        >
          <div
            className="scene-stage"
            onPointerMove={handlePointerMove}
            onPointerLeave={resetTilt}
          >
            <motion.div
              className="scene-card"
              style={{
                rotateX: imageRotateX,
                rotateY: imageRotateY,
                transformPerspective: 1200,
              }}
            >
              {/*
               * All four decoded renders stay mounted. A compositor-only CSS
               * dissolve avoids the transform/filter work that made seasonal
               * switches feel as though they travelled across the stage.
               */}
              {seasons.map((season, index) => {
                const isActive = index === activeIndex;

                return (
                  <img
                    key={season.id}
                    className={`season-image ${isActive ? "is-active" : ""}`}
                    src={`${baseUrl}${season.image}`}
                    alt={
                      isActive
                        ? `Isometric ${season.name.toLowerCase()} world generated by WorldClaw`
                        : ""
                    }
                    aria-hidden={!isActive}
                    width="1800"
                    height="900"
                    decoding="async"
                    fetchPriority={index === 0 ? "high" : "low"}
                  />
                );
              })}
            </motion.div>
          </div>

          <div
            className="season-switch"
            role="group"
            aria-label="Preview a seasonal world"
            onKeyDown={handleSeasonKeys}
            style={{ "--season-index": activeIndex } as CSSProperties}
          >
            <span className="season-switch-indicator" aria-hidden="true" />
            {seasons.map((season, index) => {
              const SeasonIcon = seasonIcons[season.id];
              const isActive = index === activeIndex;

              return (
                <button
                  type="button"
                  className={`season-chip ${isActive ? "is-active" : ""}`}
                  onClick={() => setActiveIndex(index)}
                  aria-pressed={isActive}
                  tabIndex={isActive ? 0 : -1}
                  key={season.id}
                >
                  <SeasonIcon weight="duotone" aria-hidden="true" />
                  <span>{season.name}</span>
                </button>
              );
            })}
          </div>

          <div
            className="season-copy"
            key={activeSeason.id}
            aria-live="polite"
          >
            <p className="season-asset-note">
              <span>Asset source</span>
              {activeSeason.assetNote}
            </p>
          </div>
        </motion.div>

        <div className="hero-rail">
          <dl className="hero-credits">
            {contributors.map((contributor) => (
              <div key={contributor.role}>
                <dt>{contributor.role}</dt>
                <dd>
                  {contributor.people.map((person, i) => (
                    <span key={person.name}>
                      {i > 0 && ", "}
                      {person.url ? (
                        <a
                          href={person.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {person.name}
                        </a>
                      ) : (
                        person.name
                      )}
                    </span>
                  ))}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  );
}
