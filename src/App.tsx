import { lazy, Suspense } from "react";
import { Citation } from "./components/Citation";
import { Conclusion } from "./components/Conclusion";
import { Footer } from "./components/Footer";
import { Gallery } from "./components/Gallery";
import { Hero } from "./components/Hero";
import { NavBar } from "./components/NavBar";
import { Overview } from "./components/Overview";
import { TeaserVideo } from "./components/TeaserVideo";

const Method = lazy(() =>
  import("./components/Method").then((module) => ({
    default: module.Method,
  })),
);

export default function App() {
  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <div className="grain" aria-hidden="true" />
      <NavBar />
      <main id="main-content">
        <Hero />
        <TeaserVideo />
        <Gallery />
        <Overview />
        <Suspense
          fallback={
            <section id="method" className="method section-pad" aria-busy="true">
              <div className="page-shell method-loading">
                <span />
                <span />
                <span />
                <span />
              </div>
            </section>
          }
        >
          <Method />
        </Suspense>
        <Conclusion />
        <Citation />
      </main>
      <Footer />
    </>
  );
}
