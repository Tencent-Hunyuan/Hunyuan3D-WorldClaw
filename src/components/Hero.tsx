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
  useRef,
  useState,
} from "react";
import { contributors, paperUrl, seasons } from "../data/content";
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

const ROTATE_INTERVAL = 30000;

export function Hero() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [autoRotate, setAutoRotate] = useState(true);
  const reduceMotion = useReducedMotion();
  const stageRef = useRef<HTMLDivElement>(null);
  const activeSeason = seasons[activeIndex];
  const baseUrl = import.meta.env.BASE_URL;
  const tiltX = useMotionValue(0);
  const tiltY = useMotionValue(0);
  const springX = useSpring(tiltX, { stiffness: 110, damping: 22 });
  const springY = useSpring(tiltY, { stiffness: 110, damping: 22 });
  const imageRotateX = useTransform(springX, (value) => `${value}deg`);
  const imageRotateY = useTransform(springY, (value) => `${value}deg`);

  useEffect(() => {
    if (!autoRotate || reduceMotion) return;
    const timer = window.setInterval(() => {
      if (document.hidden) return;
      setActiveIndex((index) => (index + 1) % seasons.length);
    }, ROTATE_INTERVAL);
    return () => window.clearInterval(timer);
  }, [autoRotate, reduceMotion]);

  const selectSeason = (index: number) => {
    setAutoRotate(false);
    setActiveIndex(index);
  };

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
    tiltX.set(y * -6);
    tiltY.set(x * 8);
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
    selectSeason(next);
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
          <motion.span
            key={season.id}
            style={
              {
                "--wash-soft": season.accentSoft,
                "--wash-accent": season.accent,
                "--wash-canvas": season.canvas,
              } as CSSProperties
            }
            initial={false}
            animate={{ opacity: index === activeIndex ? 1 : 0 }}
            transition={{
              duration: reduceMotion ? 0.12 : 0.6,
              ease: [0.4, 0, 0.2, 1],
            }}
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
            <p>Agentic open-world 3D scene generation at scale</p>
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
            ref={stageRef}
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
               * All four seasons stay mounted and cross-fade. Swapping the src
               * instead would stall on decoding a 1024px JPEG mid-transition.
               */}
              {seasons.map((season, index) => {
                const isActive = index === activeIndex;

                return (
                  <motion.img
                    key={season.id}
                    src={`${baseUrl}${season.image}`}
                    alt={
                      isActive
                        ? `Isometric ${season.name.toLowerCase()} world generated by WorldClaw`
                        : ""
                    }
                    aria-hidden={!isActive}
                    width="1024"
                    height="1024"
                    decoding="async"
                    fetchPriority={index === 0 ? "high" : "low"}
                    initial={false}
                    animate={{ opacity: isActive ? 1 : 0 }}
                    transition={{
                      duration: reduceMotion ? 0.12 : 0.45,
                      ease: [0.4, 0, 0.2, 1],
                    }}
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
          >
            {seasons.map((season, index) => {
              const SeasonIcon = seasonIcons[season.id];
              const isActive = index === activeIndex;

              return (
                <button
                  type="button"
                  className={`season-chip ${isActive ? "is-active" : ""}`}
                  onClick={() => selectSeason(index)}
                  aria-pressed={isActive}
                  tabIndex={isActive ? 0 : -1}
                  key={season.id}
                  style={
                    { "--chip-accent": season.accent } as CSSProperties
                  }
                >
                  <SeasonIcon weight="duotone" aria-hidden="true" />
                  <span>{season.name}</span>
                  {isActive && (
                    <motion.span
                      className="season-chip-bed"
                      layoutId="season-chip-bed"
                      transition={{
                        type: "spring",
                        stiffness: 380,
                        damping: 34,
                      }}
                      aria-hidden="true"
                    />
                  )}
                </button>
              );
            })}
          </div>

          <p className="season-line" aria-live="polite">
            {activeSeason.line}
          </p>
        </motion.div>

        <div className="hero-rail">
          <dl className="hero-credits">
            {contributors.map((contributor) => (
              <div key={contributor.role}>
                <dt>{contributor.role}</dt>
                <dd>{contributor.people}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  );
}
