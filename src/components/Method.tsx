import {
  ArrowRight,
  BoundingBox,
  Brain,
  Crosshair,
  Cube,
  ImageSquare,
  MapTrifold,
  Mountains,
  Palette,
  Quotes,
  Sparkle,
} from "@phosphor-icons/react";
import { methodStages } from "../data/content";
import { FigurePlate } from "./FigurePlate";
import { MathFormula } from "./MathFormula";
import { Reveal } from "./Reveal";
import { SectionHeader } from "./SectionHeader";

const PIPELINE_CAPTION =
  "Overview of WorldClaw. Intent analysis and planning translates the prompt into a structured scene specification; global terrain generation establishes a region-aware terrain foundation with coherent geometry, appearance, and spatial semantics; regional object generation and placement selectively populates planned regions with editable 3D assets and refines their arrangements and interactions with the terrain.";

const icons = {
  plan: Brain,
  terrain: Mountains,
  region: BoundingBox,
};

const artifactIcons = {
  regions: MapTrifold,
  terrain: Mountains,
  objects: Cube,
  layout: MapTrifold,
  assets: Cube,
  materials: Palette,
  composition: ImageSquare,
  mesh: BoundingBox,
  placement: Crosshair,
};

const composeFormula = "S = \\operatorname{Compose}(T, O)";
const edgeLabels = ["q", "P", "P + T"];

export function Method() {
  return (
    <section id="method" className="method section-pad">
      <div className="page-shell">
        <SectionHeader
          index="03"
          eyebrow="Method"
          title="Coarse to fine, global to regional."
          lede={
            <p>
              Specialized agents execute the three stages and communicate
              through shared structured intermediate representations, so local
              generation and refinement preserve the global scene organization.
            </p>
          }
        />

        <Reveal className="paper-pipeline" delay={0.08} amount={0.06}>
          <FigurePlate
            src={`${import.meta.env.BASE_URL}assets/paper/pipeline.jpg`}
            alt="Full WorldClaw pipeline from user input and intent planning through global terrain, regional object generation, refinement, and the final editable scene"
            width={3400}
            height={1687}
            label="Fig. 1"
            captionText={PIPELINE_CAPTION}
            caption={
              <span>
                Overview of WorldClaw. Intent analysis and planning translates
                the prompt into a structured scene specification; global terrain
                generation establishes a region-aware terrain foundation with
                coherent geometry, appearance, and spatial semantics; regional
                object generation and placement selectively populates planned
                regions with editable 3D assets and refines their arrangements
                and interactions with the terrain.
              </span>
            }
          />
        </Reveal>

        <Reveal className="method-flow" delay={0.08} amount={0.12}>
          <article className="method-flow-node is-terminal">
            <span className="method-node-step">Input</span>
            <div className="method-node-formula">
              <Quotes weight="light" aria-hidden="true" />
              <MathFormula formula="q" />
            </div>
            <h3>User prompt</h3>
            <span className="method-node-caption">Open-ended scene brief</span>
          </article>

          {methodStages.map((stage, index) => {
            const Icon = icons[stage.icon];

            return (
              <div className="method-flow-segment" key={stage.anchor}>
                <span className="method-connector" aria-hidden="true">
                  <MathFormula formula={edgeLabels[index]} />
                  <ArrowRight weight="light" />
                </span>
                <a className="method-flow-node is-stage" href={`#${stage.anchor}`}>
                  <span className="method-node-step">
                    Stage {String(index + 1).padStart(2, "0")}
                  </span>
                  <div className="method-node-formula">
                    <Icon weight="light" aria-hidden="true" />
                    <MathFormula formula={stage.formula} />
                  </div>
                  <h3>{stage.title}</h3>
                  <span className="method-node-caption">
                    {stage.output.label}
                    <MathFormula formula={stage.output.formula} />
                  </span>
                </a>
              </div>
            );
          })}

          <span className="method-connector" aria-hidden="true">
            <MathFormula formula="T + O" />
            <ArrowRight weight="light" />
          </span>
          <article className="method-flow-node is-terminal is-output">
            <span className="method-node-step">Output</span>
            <div className="method-node-formula">
              <Sparkle weight="light" aria-hidden="true" />
              <MathFormula formula={composeFormula} />
            </div>
            <h3>Composed world</h3>
            <span className="method-node-caption">
              Explicit, explorable, editable
            </span>
          </article>
        </Reveal>

        <div className="method-details">
          {methodStages.map((stage, index) => {
            const Icon = icons[stage.icon];

            return (
              <Reveal
                className="method-detail"
                delay={index * 0.04}
                amount={0.12}
                key={stage.anchor}
              >
                <article id={stage.anchor}>
                  <div className="method-detail-meta">
                    <span className="method-detail-step">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div className="method-detail-formula">
                      <Icon weight="light" aria-hidden="true" />
                      <MathFormula formula={stage.formula} />
                    </div>
                    <dl className="method-detail-io">
                      <div>
                        <dt>In</dt>
                        <dd>
                          <span>{stage.input.label}</span>
                          <MathFormula formula={stage.input.formula} />
                        </dd>
                      </div>
                      <div>
                        <dt>Out</dt>
                        <dd>
                          <span>{stage.output.label}</span>
                          <MathFormula formula={stage.output.formula} />
                        </dd>
                      </div>
                    </dl>
                  </div>

                  <div className="method-detail-copy">
                    <h3>{stage.title}</h3>
                    <p>{stage.description}</p>

                    <div className="method-artifacts">
                      {stage.artifacts.map((artifact) => {
                        const ArtifactIcon = artifactIcons[artifact.icon];

                        return (
                          <div className="method-artifact" key={artifact.symbol}>
                            <div className="method-artifact-symbol">
                              <ArtifactIcon weight="light" aria-hidden="true" />
                              <MathFormula formula={artifact.symbol} />
                            </div>
                            <strong>{artifact.title}</strong>
                            <span>{artifact.detail}</span>
                          </div>
                        );
                      })}
                    </div>

                    <div className="method-equation">
                      <MathFormula formula={stage.equation.formula} displayMode />
                      <p>{stage.equation.caption}</p>
                    </div>

                    {"figure" in stage && stage.figure ? (
                      <FigurePlate
                        className="method-detail-figure"
                        src={`${import.meta.env.BASE_URL}${stage.figure.src}`}
                        alt={stage.figure.alt}
                        width={stage.figure.width}
                        height={stage.figure.height}
                        label={stage.figure.label}
                        caption={stage.figure.caption}
                        captionText={stage.figure.caption}
                      />
                    ) : null}
                  </div>
                </article>
              </Reveal>
            );
          })}
        </div>

      </div>
    </section>
  );
}
