/* ============================================================
   REAL-TIME NETWORK INTRUSION DETECTION
   DASHBOARD DATA BINDING
   ============================================================

   Backend source:
       /api/dashboard

   Expected normalized structure:

   {
       system,
       anomalies,
       drift,
       retraining,
       retraining_history,
       models,
       model_metadata,
       top_model,
       version_management,
       pipeline,
       self_correction,
       pipeline_log,
       evaluation,
       summary,
       timestamp
   }

   IMPORTANT:
   This file is bound to the CURRENT dashboard.html IDs.
   ============================================================ */


"use strict";


// ============================================================
// CONFIGURATION
// ============================================================

const REFRESH_INTERVAL = 5000;

let dashboardRefreshTimer = null;
let lastDashboardData = null;



// ============================================================
// DOM HELPERS
// ============================================================

function $(id) {
    return document.getElementById(id);
}


function setText(id, value) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.textContent =
        value === null ||
        value === undefined ||
        value === ""
            ? "—"
            : String(value);
}


function setHTML(id, html) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.innerHTML = html;
}


function safe(value, fallback = "—") {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return fallback;
    }

    return value;
}


function firstDefined(...values) {

    for (const value of values) {

        if (
            value !== null &&
            value !== undefined &&
            value !== ""
        ) {
            return value;
        }
    }

    return null;
}


function numberValue(value, fallback = null) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return fallback;
    }

    const number =
        Number(value);

    return Number.isFinite(number)
        ? number
        : fallback;
}


function integer(value, fallback = "—") {

    const number =
        numberValue(value);

    if (number === null) {
        return fallback;
    }

    return Math.round(number)
        .toLocaleString();
}


function decimal(
    value,
    digits = 4,
    fallback = "—"
) {

    const number =
        numberValue(value);

    if (number === null) {
        return fallback;
    }

    return number.toFixed(digits);
}


function percentage(
    value,
    digits = 2
) {

    const number =
        numberValue(value);

    if (number === null) {
        return "—";
    }

    return `${(
        number * 100
    ).toFixed(digits)}%`;
}


function escapeHTML(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function formatTimestamp(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }

    /*
     * Numeric UNIX timestamp.
     */
    if (
        typeof value === "number" ||
        (
            typeof value === "string" &&
            /^-?\d+(\.\d+)?$/.test(value)
        )
    ) {

        let number =
            Number(value);

        /*
         * Python time.time() is seconds.
         * JavaScript Date expects milliseconds.
         */
        if (
            Math.abs(number) < 100000000000
        ) {
            number *= 1000;
        }

        const date =
            new Date(number);

        if (
            !Number.isNaN(
                date.getTime()
            )
        ) {
            return date.toLocaleString();
        }
    }


    /*
     * Normal date string.
     */
    const date =
        new Date(value);

    if (
        !Number.isNaN(
            date.getTime()
        )
    ) {
        return date.toLocaleString();
    }

    return String(value);
}


// ============================================================
// STATUS HELPERS
// ============================================================

function normalizeStatus(value) {

    return String(
        safe(value, "WAITING")
    )
        .trim()
        .toUpperCase();
}


function statusClass(status) {

    const value =
        normalizeStatus(status);


    if (
        value.includes("ERROR") ||
        value.includes("FAILED") ||
        value.includes("REJECT") ||
        value.includes("STOPPED")
    ) {
        return "badge-danger";
    }


    if (
        value.includes("COMPLETED") ||
        value.includes("PROMOTED") ||
        value.includes("PROMOTE") ||
        value.includes("PASS") ||
        value === "ACTIVE" ||
        value === "RUNNING" ||
        value === "DETECTED"
    ) {
        return "badge-success";
    }


    if (
        value.includes("TRIGGER") ||
        value.includes("TRAIN") ||
        value.includes("DRIFT") ||
        value.includes("PROCESS") ||
        value.includes("ONGOING")
    ) {
        return "badge-warning";
    }


    return "badge-neutral";
}


function statusBadge(status) {

    const value =
        normalizeStatus(status);

    return `
        <span class="badge ${statusClass(value)}">
            ${escapeHTML(value)}
        </span>
    `;
}


// ============================================================
// PIPELINE STAGE HELPERS
// ============================================================

function setPipelineStage(
    stageId,
    status
) {

    const element =
        $(stageId);

    if (!element) {
        return;
    }

    element.classList.remove(
        "completed",
        "running",
        "failed",
        "rejected",
        "waiting",
        "active"
    );

    element.classList.add(
        status
    );
}


function stageText(
    elementId,
    status
) {

    setText(
        elementId,
        normalizeStatus(status)
    );
}


// ============================================================
// API
// ============================================================

async function getDashboard() {

    const response =
        await fetch(
            "/api/dashboard",
            {
                method: "GET",
                cache: "no-store",
                headers: {
                    "Accept":
                        "application/json"
                }
            }
        );


    if (!response.ok) {

        throw new Error(
            `Dashboard API returned HTTP ${response.status}`
        );
    }


    const data =
        await response.json();


    if (
        !data ||
        typeof data !== "object"
    ) {

        throw new Error(
            "Dashboard API returned invalid JSON."
        );
    }


    return data;
}


// ============================================================
// SYSTEM OVERVIEW
// ============================================================

function renderSystem(data) {

    const system =
        data.system || {};

    const summary =
        data.summary || {};

    const version =
        data.version_management || {};


    /*
     * IMPORTANT:
     * These are the ACTUAL backend field names.
     *
     * system.total_logs
     * system.logs_processed
     * system.normal_logs
     * system.anomalies_detected
     */

    const totalLogs =
        firstDefined(
            system.total_logs,
            summary.total_logs
        );


    const processedLogs =
        firstDefined(
            system.logs_processed,
            summary.processed
        );


    const normalLogs =
        firstDefined(
            system.normal_logs,
            summary.normal
        );


    const anomalyCount =
        firstDefined(
            system.anomalies_detected,
            summary.anomalies
        );


    setText(
        "totalLogs",
        integer(totalLogs)
    );


    setText(
        "processedLogs",
        integer(processedLogs)
    );


    setText(
        "normalLogs",
        integer(normalLogs)
    );


    setText(
        "anomalyCount",
        integer(anomalyCount)
    );


    /*
     * Current model versions.
     */
    const productionVersion =
        firstDefined(
            version.current_production_version,
            system.production_version,
            summary.production_version
        );


    const candidateVersion =
        firstDefined(
            version.latest_candidate_version,
            system.candidate_version,
            summary.candidate_version
        );


    const decision =
        firstDefined(
            version.last_decision,
            system.last_decision,
            summary.last_decision
        );


    /*
     * Current HTML has no dedicated version KPI,
     * but model/version sections use these values.
     */
    setText(
        "metadataProduction",
        productionVersion !== null
            ? `v${productionVersion}`
            : "—"
    );


    /*
     * System status.
     */
    const systemStatus =
        firstDefined(
            system.status,
            system.system_status,
            "RUNNING"
        );


    setText(
        "systemStatus",
        normalizeStatus(systemStatus)
    );


    const dot =
        $("systemDot");


    if (dot) {

        dot.classList.remove(
            "online",
            "offline",
            "warning"
        );


        const normalized =
            normalizeStatus(systemStatus);


        if (
            normalized.includes("RUN") ||
            normalized.includes("ACTIVE") ||
            normalized.includes("ONLINE")
        ) {

            dot.classList.add(
                "online"
            );
        }

        else if (
            normalized.includes("ERROR") ||
            normalized.includes("STOP") ||
            normalized.includes("FAIL")
        ) {

            dot.classList.add(
                "offline"
            );
        }

        else {

            dot.classList.add(
                "warning"
            );
        }
    }


    setText(
        "systemTimestamp",
        formatTimestamp(
            firstDefined(
                system.timestamp,
                system.last_updated,
                data.timestamp
            )
        )
    );
}


// ============================================================
// PIPELINE
// ============================================================

function renderPipeline(data) {

    const pipeline =
        data.pipeline || {};

    const summary =
        data.summary || {};

    const drift =
        data.drift || {};

    const retraining =
        data.retraining || {};

    const evaluation =
        data.evaluation || {};

    const version =
        data.version_management || {};

    const selfCorrection =
        data.self_correction || {};


    /*
     * --------------------------------------------------------
     * CURRENT RECORD
     * --------------------------------------------------------
     */

    const currentRecord =
        firstDefined(
            pipeline.latest_record,
            pipeline.current_record,
            summary.latest_record
        );


    /*
     * --------------------------------------------------------
     * CHECKPOINT
     * --------------------------------------------------------
     */

    const checkpoint =
        firstDefined(
            pipeline.checkpoint,
            pipeline.last_checkpoint,
            summary.last_checkpoint
        );


    /*
     * --------------------------------------------------------
     * NEXT CHECKPOINT
     * --------------------------------------------------------
     */

    const nextCheckpoint =
        firstDefined(
            pipeline.next_checkpoint,
            summary.next_checkpoint
        );


    /*
     * --------------------------------------------------------
     * DRIFT
     * --------------------------------------------------------
     */

    const driftCount =
        firstDefined(
            drift.total_drifts,
            pipeline.drift_count,
            summary.total_drifts,
            0
        );


    /*
     * --------------------------------------------------------
     * RETRAINING STATUS
     * --------------------------------------------------------
     */

    let retrainingStatus =
        firstDefined(
            pipeline.retraining_status,
            pipeline.retraining,
            retraining.status,
            retraining.latest &&
                retraining.latest.status
        );


    if (
        retrainingStatus === null
    ) {

        retrainingStatus =
            retraining.available
                ? "COMPLETED"
                : "NOT_TRIGGERED";
    }


    /*
     * --------------------------------------------------------
     * EVALUATION STATUS
     * --------------------------------------------------------
     */

    const evaluationDecision =
        firstDefined(
            evaluation.decision,
            evaluation.metadata &&
                evaluation.metadata.last_decision,
            version.last_decision
        );


    const evaluationStatus =
        firstDefined(
            pipeline.evaluation_status,
            pipeline.evaluation,
            evaluation.status,
            evaluationDecision,
            "WAITING"
        );


    /*
     * --------------------------------------------------------
     * LIFECYCLE STATUS
     * --------------------------------------------------------
     */

    const lifecycleStatus =
        firstDefined(
            pipeline.lifecycle_status,
            pipeline.lifecycle,
            pipeline.final_result,
            version.last_decision,
            "WAITING"
        );


    /*
     * --------------------------------------------------------
     * PIPELINE GLOBAL STATUS
     * --------------------------------------------------------
     */

    let pipelineStatus =
        firstDefined(
            pipeline.status,
            pipeline.pipeline_status
        );


    if (
        pipelineStatus === null
    ) {

        if (
            normalizeStatus(lifecycleStatus)
                .includes("REJECT")
        ) {
            pipelineStatus =
                "COMPLETED";
        }

        else if (
            retraining.available
        ) {
            pipelineStatus =
                "COMPLETED";
        }

        else {
            pipelineStatus =
                "RUNNING";
        }
    }


    /*
     * --------------------------------------------------------
     * HTML BINDINGS
     * --------------------------------------------------------
     */

    setHTML(
        "pipelineStatus",
        statusBadge(
            pipelineStatus
        )
    );


    setText(
        "pipelineCurrentRecord",
        integer(currentRecord)
    );


    setText(
        "pipelineCheckpoint",
        integer(checkpoint)
    );


    setText(
        "pipelineNextCheckpoint",
        integer(nextCheckpoint)
    );


    setText(
        "pipelineDriftCount",
        integer(driftCount)
    );


    setText(
        "pipelineRetrainingStatus",
        normalizeStatus(
            retrainingStatus
        )
    );


    setText(
        "pipelineLifecycleDecision",
        normalizeStatus(
            lifecycleStatus
        )
    );


    /*
     * --------------------------------------------------------
     * STAGE 01 - STREAM
     * --------------------------------------------------------
     */

    const streamState =
        numberValue(
            currentRecord,
            0
        ) > 0
            ? "completed"
            : "waiting";


    setPipelineStage(
        "stageStream",
        streamState
    );


    /*
     * --------------------------------------------------------
     * STAGE 02 - ADWIN
     * --------------------------------------------------------
     */

    const normalizedDrift =
        normalizeStatus(
            driftCount > 0
                ? "DRIFT DETECTED"
                : "NO DRIFT"
        );


    stageText(
        "pipelineAdwinStatus",
        normalizedDrift
    );


    setPipelineStage(
        "stageDrift",
        numberValue(
            driftCount,
            0
        ) > 0
            ? "completed"
            : "waiting"
    );


    /*
     * --------------------------------------------------------
     * STAGE 03 - RETRAINING
     * --------------------------------------------------------
     */

    stageText(
        "pipelineRetrainStatus",
        retrainingStatus
    );


    const normalizedRetraining =
        normalizeStatus(
            retrainingStatus
        );


    let retrainingStage =
        "waiting";


    if (
        normalizedRetraining.includes("RUN") ||
        normalizedRetraining.includes("TRAIN") ||
        normalizedRetraining.includes("ONGOING") ||
        normalizedRetraining.includes("PROCESS")
    ) {

        retrainingStage =
            "running";
    }

    else if (
        normalizedRetraining.includes("COMPLETED") ||
        normalizedRetraining.includes("TRIGGERED")
    ) {

        retrainingStage =
            "completed";
    }

    else if (
        normalizedRetraining.includes("FAILED") ||
        normalizedRetraining.includes("ERROR")
    ) {

        retrainingStage =
            "failed";
    }


    setPipelineStage(
        "stageRetrain",
        retrainingStage
    );


    /*
     * --------------------------------------------------------
     * STAGE 04 - EVALUATION
     * --------------------------------------------------------
     */

    stageText(
        "pipelineEvaluationStatus",
        evaluationStatus
    );


    const normalizedEvaluation =
        normalizeStatus(
            evaluationStatus
        );


    let evaluationStage =
        "waiting";


    if (
        normalizedEvaluation.includes("FAIL") ||
        normalizedEvaluation.includes("ERROR")
    ) {

        evaluationStage =
            "failed";
    }

    else if (
        normalizedEvaluation.includes("REJECT")
    ) {

        evaluationStage =
            "rejected";
    }

    else if (
        normalizedEvaluation.includes("COMPLETED") ||
        normalizedEvaluation.includes("PASS") ||
        normalizedEvaluation.includes("PROMOT")
    ) {

        evaluationStage =
            "completed";
    }

    else if (
        normalizedEvaluation.includes("RUN") ||
        normalizedEvaluation.includes("EVALUAT")
    ) {

        evaluationStage =
            "running";
    }


    setPipelineStage(
        "stageEvaluation",
        evaluationStage
    );


    /*
     * --------------------------------------------------------
     * STAGE 05 - LIFECYCLE
     * --------------------------------------------------------
     */

    stageText(
        "pipelineLifecycleStatus",
        lifecycleStatus
    );


    const normalizedLifecycle =
        normalizeStatus(
            lifecycleStatus
        );


    let lifecycleStage =
        "waiting";


    if (
        normalizedLifecycle.includes("REJECT")
    ) {

        lifecycleStage =
            "rejected";
    }

    else if (
        normalizedLifecycle.includes("PROMOT") ||
        normalizedLifecycle.includes("ACCEPT") ||
        normalizedLifecycle.includes("ACTIVE") ||
        normalizedLifecycle.includes("COMPLETED")
    ) {

        lifecycleStage =
            "completed";
    }

    else if (
        normalizedLifecycle.includes("RUN")
    ) {

        lifecycleStage =
            "running";
    }


    setPipelineStage(
        "stageLifecycle",
        lifecycleStage
    );


    /*
     * --------------------------------------------------------
     * PIPELINE MESSAGE
     * --------------------------------------------------------
     */

    const message =
        firstDefined(
            pipeline.message,
            pipeline.current_operation,
            pipeline.stage_message,
            selfCorrection.message,
            selfCorrection.current_operation,
            pipeline.display_result,
            version.last_decision,
            "Monitoring stream..."
        );


    setText(
        "pipelineMessage",
        message
    );
}


// ============================================================
// DRIFT
// ============================================================

function renderDrift(data) {

    const drift =
        data.drift || {};

    const records =
        Array.isArray(
            drift.records
        )
            ? drift.records
            : [];


    const latest =
        drift.latest_drift ||
        drift.latest ||
        (
            records.length
                ? records[
                    records.length - 1
                ]
                : null
        );


    /*
     * --------------------------------------------------------
     * SUMMARY
     * --------------------------------------------------------
     */

    setText(
        "totalDrifts",
        integer(
            firstDefined(
                drift.total_drifts,
                records.length,
                0
            )
        )
    );


    setText(
        "newDriftCount",
        integer(
            firstDefined(
                drift.new_drift_count,
                0
            )
        )
    );


    /*
     * Latest drift record.
     */

    if (latest) {

        setText(
            "latestDriftRecord",
            integer(
                firstDefined(
                    latest.record_index,
                    latest.record
                )
            )
        );


        setText(
            "driftCheckpoint",
            integer(
                firstDefined(
                    latest.checkpoint,
                    latest.last_checkpoint
                )
            )
        );


        const latestContainer =
            $("latestDrift");


        if (latestContainer) {

            latestContainer.innerHTML = `
                <div class="event-grid">

                    <div>
                        <span>TIME</span>
                        <strong>
                            ${escapeHTML(
                                formatTimestamp(
                                    firstDefined(
                                        latest.timestamp,
                                        latest.detected_at,
                                        latest.time
                                    )
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>RECORD</span>
                        <strong>
                            ${escapeHTML(
                                integer(
                                    firstDefined(
                                        latest.record_index,
                                        latest.record
                                    )
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>SCORE</span>
                        <strong>
                            ${escapeHTML(
                                decimal(
                                    firstDefined(
                                        latest.final_score,
                                        latest.score
                                    ),
                                    4
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>ADWIN</span>
                        <strong>
                            ${escapeHTML(
                                decimal(
                                    firstDefined(
                                        latest.adwin_estimation,
                                        latest.adwin
                                    ),
                                    4
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>WINDOW WIDTH</span>
                        <strong>
                            ${escapeHTML(
                                integer(
                                    firstDefined(
                                        latest.adwin_width,
                                        latest.width
                                    )
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>STATUS</span>
                        <strong>
                            ${statusBadge(
                                "DRIFT DETECTED"
                            )}
                        </strong>
                    </div>

                </div>
            `;
        }
    }

    else {

        setHTML(
            "latestDrift",
            `
                <div class="empty-state">
                    No drift event loaded.
                </div>
            `
        );
    }


    /*
     * Badge.
     */

    setHTML(
        "driftBadge",
        statusBadge(
            numberValue(
                drift.total_drifts,
                0
            ) > 0
                ? "DETECTED"
                : "WAITING"
        )
    );


    /*
     * History.
     */

    renderDriftHistory(
        records
    );
}


// ============================================================
// DRIFT HISTORY
// ============================================================

function renderDriftHistory(records) {

    const container =
        $("driftHistory");


    if (!container) {
        return;
    }


    if (!records.length) {

        container.innerHTML = `
            <div class="empty-state">
                No drift events recorded.
            </div>
        `;

        return;
    }


    const ordered =
        records
            .slice()
            .reverse();


    container.innerHTML =
        ordered
            .map(
                (
                    record,
                    reverseIndex
                ) => {

                    const eventNumber =
                        records.length -
                        reverseIndex;


                    return `
                        <div class="drift-event">

                            <div class="drift-event-header">

                                <strong>
                                    Drift #${eventNumber}
                                </strong>

                                <span>
                                    ${escapeHTML(
                                        formatTimestamp(
                                            firstDefined(
                                                record.timestamp,
                                                record.detected_at,
                                                record.time
                                            )
                                        )
                                    )}
                                </span>

                            </div>

                            <div class="drift-event-grid">

                                <div>
                                    <span>RECORD</span>
                                    <strong>
                                        ${escapeHTML(
                                            integer(
                                                firstDefined(
                                                    record.record_index,
                                                    record.record
                                                )
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>SCORE</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                firstDefined(
                                                    record.final_score,
                                                    record.score
                                                ),
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>ADWIN</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                firstDefined(
                                                    record.adwin_estimation,
                                                    record.adwin
                                                ),
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>WIDTH</span>
                                    <strong>
                                        ${escapeHTML(
                                            integer(
                                                firstDefined(
                                                    record.adwin_width,
                                                    record.width
                                                )
                                            )
                                        )}
                                    </strong>
                                </div>

                            </div>

                        </div>
                    `;
                }
            )
            .join("");
}


// ============================================================
// RETRAINING
// ============================================================

function renderRetraining(data) {

    const retraining =
        data.retraining || {};

    const historyData =
        data.retraining_history || {};


    const records =
        Array.isArray(
            retraining.records
        )
            ? retraining.records
            : [];


    const latest =
        retraining.latest ||
        (
            records.length
                ? records[
                    records.length - 1
                ]
                : null
        );


    const available =
        retraining.available === true ||
        records.length > 0;


    /*
     * Status.
     */

    let status =
        firstDefined(
            retraining.status,
            latest && latest.status
        );


    if (
        status === null
    ) {

        status =
            available
                ? "COMPLETED"
                : "NOT_TRIGGERED";
    }


    setText(
        "retrainingStatus",
        normalizeStatus(status)
    );


    setHTML(
        "retrainingBadge",
        statusBadge(status)
    );


    setText(
        "retrainingStatusMessage",
        firstDefined(
            retraining.message,
            latest && latest.message,
            available
                ? "Latest retraining result loaded."
                : "Waiting for ADWIN drift trigger."
        )
    );


    /*
     * Result container.
     */

    const container =
        $("retrainingContainer");


    if (container) {

        if (!latest) {

            container.innerHTML = `
                <div class="empty-state">
                    No retraining result available.
                </div>
            `;
        }

        else {

            const model =
                firstDefined(
                    latest.model,
                    latest.model_name,
                    "MODEL"
                );


            const modelType =
                firstDefined(
                    latest.model_type,
                    "candidate"
                );


            const trainingRecords =
                firstDefined(
                    latest.training_records,
                    latest.records
                );


            const features =
                firstDefined(
                    latest.features,
                    latest.feature_count
                );


            const trainingTime =
                firstDefined(
                    latest.training_time_seconds,
                    latest.training_time
                );


            const epochs =
                firstDefined(
                    latest.epochs,
                    latest.ae_epochs
                );


            const threshold =
                firstDefined(
                    latest.ae_threshold,
                    latest.threshold
                );


            container.innerHTML = `
                <div class="training-result">

                    <div class="training-result-grid">

                        <div>
                            <span>MODEL</span>
                            <strong>
                                ${escapeHTML(
                                    model
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>TYPE</span>
                            <strong>
                                ${escapeHTML(
                                    modelType
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>RECORDS</span>
                            <strong>
                                ${escapeHTML(
                                    integer(
                                        trainingRecords
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>FEATURES</span>
                            <strong>
                                ${escapeHTML(
                                    integer(
                                        features
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>TRAINING TIME</span>
                            <strong>
                                ${escapeHTML(
                                    trainingTime !== null
                                        ? `${trainingTime}s`
                                        : "—"
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>EPOCHS</span>
                            <strong>
                                ${escapeHTML(
                                    integer(
                                        epochs
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>AE THRESHOLD</span>
                            <strong>
                                ${escapeHTML(
                                    decimal(
                                        threshold,
                                        6
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>STATUS</span>
                            <strong>
                                ${statusBadge(
                                    status
                                )}
                            </strong>
                        </div>

                    </div>

                </div>
            `;
        }
    }


    /*
     * History.
     */

    const historyRecords =
        Array.isArray(
            historyData.records
        )
            ? historyData.records
            : records;


    renderRetrainingHistory(
        historyRecords
    );
}


// ============================================================
// RETRAINING HISTORY
// ============================================================

function renderRetrainingHistory(records) {

    const container =
        $("retrainingHistory");


    if (!container) {
        return;
    }


    if (!records.length) {

        container.innerHTML = `
            <div class="empty-state">
                No retraining history available.
            </div>
        `;

        return;
    }


    container.innerHTML =
        records
            .slice()
            .reverse()
            .map(
                record => {

                    const model =
                        firstDefined(
                            record.model,
                            record.model_name,
                            "MODEL"
                        );


                    const modelType =
                        firstDefined(
                            record.model_type,
                            "candidate"
                        );


                    const recordsCount =
                        firstDefined(
                            record.training_records,
                            record.records
                        );


                    const features =
                        firstDefined(
                            record.features,
                            record.feature_count
                        );


                    const trainingTime =
                        firstDefined(
                            record.training_time_seconds,
                            record.training_time
                        );


                    const status =
                        firstDefined(
                            record.status,
                            "UNKNOWN"
                        );


                    return `
                        <div class="training-history-row">

                            <div>
                                <strong>
                                    ${escapeHTML(
                                        model
                                    )}
                                </strong>

                                <small>
                                    ${escapeHTML(
                                        modelType
                                    )}
                                </small>
                            </div>

                            <div>
                                <span>Records</span>
                                <strong>
                                    ${escapeHTML(
                                        integer(
                                            recordsCount
                                        )
                                    )}
                                </strong>
                            </div>

                            <div>
                                <span>Features</span>
                                <strong>
                                    ${escapeHTML(
                                        integer(
                                            features
                                        )
                                    )}
                                </strong>
                            </div>

                            <div>
                                <span>Time</span>
                                <strong>
                                    ${escapeHTML(
                                        trainingTime !== null
                                            ? `${trainingTime}s`
                                            : "—"
                                    )}
                                </strong>
                            </div>

                            <div>
                                ${statusBadge(
                                    status
                                )}
                            </div>

                        </div>
                    `;
                }
            )
            .join("");
}


// ============================================================
// MODEL METADATA
// ============================================================

function renderModelMetadata(data) {

    const metadata =
        data.model_metadata || {};

    const models =
        Array.isArray(
            metadata.models
        )
            ? metadata.models
            : [];


    const production =
        metadata.production || {};


    const candidate =
        metadata.candidate || {};


    const scoreConfig =
        metadata.score_configuration || {};


    /*
     * Production version.
     */

    const versionData =
        data.version_management || {};


    const productionVersion =
        firstDefined(
            versionData.current_production_version,
            data.models &&
                data.models.current_production_version
        );


    setText(
        "metadataProduction",
        productionVersion !== null
            ? `v${productionVersion}`
            : "—"
    );


    /*
     * Container.
     */

    const container =
        $("modelMetadataContainer");


    if (!container) {
        return;
    }


    const entries = [];


    if (
        Object.keys(production).length
    ) {

        entries.push({
            title:
                "Production Model",
            data:
                production
        });
    }


    if (
        Object.keys(candidate).length
    ) {

        entries.push({
            title:
                "Candidate Model",
            data:
                candidate
        });
    }


    /*
     * If production/candidate aren't available,
     * use models[].
     */

    if (
        !entries.length &&
        models.length
    ) {

        models.forEach(
            (model, index) => {

                entries.push({
                    title:
                        `Model ${index + 1}`,
                    data:
                        model
                });
            }
        );
    }


    if (!entries.length) {

        container.innerHTML = `
            <div class="empty-state">
                No model metadata available.
            </div>
        `;

        return;
    }


    let html = "";


    entries.forEach(
        entry => {

            const model =
                entry.data || {};


            html += `
                <div class="model-card">

                    <div class="model-card-header">

                        <div>
                            <strong>
                                ${escapeHTML(
                                    entry.title
                                )}
                            </strong>

                            <small>
                                ${escapeHTML(
                                    firstDefined(
                                        model.model_type,
                                        model.type,
                                        "MODEL"
                                    )
                                )}
                            </small>
                        </div>

                    </div>

                    <div class="model-metadata-grid">

                        <div>
                            <span>VERSION</span>
                            <strong>
                                ${escapeHTML(
                                    firstDefined(
                                        model.version
                                    ) !== null
                                        ? `v${model.version}`
                                        : "—"
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>F1</span>
                            <strong>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            model.f1,
                                            model.metrics &&
                                                model.metrics.f1
                                        ),
                                        4
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>ACCURACY</span>
                            <strong>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            model.accuracy,
                                            model.metrics &&
                                                model.metrics.accuracy
                                        ),
                                        4
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>DETECTION RATE</span>
                            <strong>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            model.detection_rate,
                                            model.metrics &&
                                                model.metrics.detection_rate
                                        ),
                                        4
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>FAR</span>
                            <strong>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            model.false_alarm_rate,
                                            model.far,
                                            model.metrics &&
                                                model.metrics.false_alarm_rate
                                        ),
                                        4
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>STATUS</span>
                            <strong>
                                ${statusBadge(
                                    firstDefined(
                                        model.status,
                                        entry.title
                                    )
                                )}
                            </strong>
                        </div>

                    </div>

                </div>
            `;
        }
    );


    /*
     * Score configuration.
     */

    html += `
        <div class="model-card">

            <div class="model-card-header">

                <div>
                    <strong>
                        Fusion Configuration
                    </strong>

                    <small>
                        IF + AE scoring
                    </small>
                </div>

            </div>

            <div class="model-metadata-grid">

                <div>
                    <span>IF WEIGHT</span>
                    <strong>
                        ${escapeHTML(
                            decimal(
                                scoreConfig.if_weight,
                                4
                            )
                        )}
                    </strong>
                </div>

                <div>
                    <span>AE WEIGHT</span>
                    <strong>
                        ${escapeHTML(
                            decimal(
                                scoreConfig.ae_weight,
                                4
                            )
                        )}
                    </strong>
                </div>

                <div>
                    <span>FUSION THRESHOLD</span>
                    <strong>
                        ${escapeHTML(
                            decimal(
                                scoreConfig.fusion_threshold,
                                4
                            )
                        )}
                    </strong>
                </div>

                <div>
                    <span>F1 TOLERANCE</span>
                    <strong>
                        ${escapeHTML(
                            decimal(
                                scoreConfig.f1_tolerance,
                                4
                            )
                        )}
                    </strong>
                </div>

                <div>
                    <span>FAR TOLERANCE</span>
                    <strong>
                        ${escapeHTML(
                            decimal(
                                scoreConfig.false_alarm_tolerance,
                                4
                            )
                        )}
                    </strong>
                </div>

                <div>
                    <span>EVALUATION RECORDS</span>
                    <strong>
                        ${escapeHTML(
                            integer(
                                scoreConfig.evaluation_records
                            )
                        )}
                    </strong>
                </div>

            </div>

        </div>
    `;


    container.innerHTML =
        html;
}


/* ============================================================
   TOP PERFORMING MODEL
   ============================================================ */

function renderTopModel(data) {

    const topData =
        data.top_model || {};

    const model =
        topData.model && typeof topData.model === "object"
            ? topData.model
            : topData;

    const container =
        document.getElementById(
            "topModelContainer"
        );

    const badge =
        document.getElementById(
            "topModelBadge"
        );


    /*
     * No model available
     */

    if (
        !topData.available ||
        !model ||
        !model.model
    ) {

        if (badge) {
            badge.textContent = "—";
        }

        if (container) {
            container.innerHTML = `
                <div class="empty-state">
                    No evaluated model is available.
                </div>
            `;
        }

        return;
    }


    /*
     * Backend values
     */

    const modelName =
        model.model ||
        model.model_name ||
        "—";

    const f1 =
        model.f1;

    const accuracy =
        model.accuracy;

    const detection =
        model.detection_rate;

    const far =
        model.false_alarm_rate;

    const reason =
        topData.selection_reason ||
        "Highest F1 among evaluated model versions.";


    /*
     * TOP MODEL BADGE
     */

    if (badge) {

        badge.textContent =
            "TOP MODEL";

        badge.className =
            "badge badge-neutral";
    }


    /*
     * RENDER THE COMPLETE CARD
     *
     * IMPORTANT:
     * The current HTML only gives us
     * #topModelContainer, so we create
     * the value elements here.
     */

    if (container) {

        container.innerHTML = `

            <div class="top-model-card">

                <div class="top-model-header">

                    <div>
                        <strong>
                            ${modelName}
                        </strong>

                        <span>
                            Evaluated model
                        </span>
                    </div>

                    <div class="badge badge-neutral">
                        TOP MODEL
                    </div>

                </div>


                <div class="top-model-metrics">

                    <div class="top-model-metric">

                        <span>
                            F1
                        </span>

                        <strong>
                            ${
                                f1 !== null &&
                                f1 !== undefined
                                    ? Number(f1).toFixed(4)
                                    : "—"
                            }
                        </strong>

                    </div>


                    <div class="top-model-metric">

                        <span>
                            ACCURACY
                        </span>

                        <strong>
                            ${
                                accuracy !== null &&
                                accuracy !== undefined
                                    ? Number(accuracy).toFixed(4)
                                    : "—"
                            }
                        </strong>

                    </div>


                    <div class="top-model-metric">

                        <span>
                            DETECTION
                        </span>

                        <strong>
                            ${
                                detection !== null &&
                                detection !== undefined
                                    ? Number(detection).toFixed(4)
                                    : "—"
                            }
                        </strong>

                    </div>


                    <div class="top-model-metric">

                        <span>
                            FAR
                        </span>

                        <strong>
                            ${
                                far !== null &&
                                far !== undefined
                                    ? Number(far).toFixed(4)
                                    : "—"
                            }
                        </strong>

                    </div>

                </div>


                <div class="top-model-reason">

                    <span>
                        SELECTION REASON
                    </span>

                    <strong>
                        ${reason}
                    </strong>

                </div>

            </div>
        `;
    }
}


// ============================================================
// MODEL VERSION MANAGEMENT
// ============================================================

function renderModels(data) {

    const versionData =
        data.version_management || {};

    const modelsData =
        data.models || {};


    const models =
        Array.isArray(
            modelsData.models
        )
            ? modelsData.models
            : (
                Array.isArray(
                    versionData.versions
                )
                    ? versionData.versions
                    : []
            );


    const production =
        firstDefined(
            versionData.current_production_version,
            modelsData.current_production_version
        );


    const candidate =
        firstDefined(
            versionData.latest_candidate_version,
            modelsData.latest_candidate_version
        );


    const decision =
        firstDefined(
            versionData.last_decision,
            modelsData.last_decision
        );


    const container =
        $("modelsContainer");


    if (!container) {
        return;
    }


    if (!models.length) {

        container.innerHTML = `
            <div class="empty-state">
                No model versions registered.
            </div>
        `;

        setHTML(
            "modelDecision",
            statusBadge(
                decision || "WAITING"
            )
        );

        return;
    }


    container.innerHTML =
        models
            .slice()
            .reverse()
            .map(
                model => {

                    const version =
                        firstDefined(
                            model.version
                        );


                    const modelType =
                        firstDefined(
                            model.model_type,
                            model.type,
                            "MODEL"
                        );


                    const metrics =
                        model.metrics || {};


                    let status =
                        firstDefined(
                            model.status
                        );


                    if (
                        status === null
                    ) {

                        if (
                            version !== null &&
                            production !== null &&
                            Number(version) ===
                                Number(production)
                        ) {

                            status =
                                "PRODUCTION";
                        }

                        else if (
                            version !== null &&
                            candidate !== null &&
                            Number(version) ===
                                Number(candidate)
                        ) {

                            status =
                                "CANDIDATE";
                        }

                        else {

                            status =
                                "ARCHIVED";
                        }
                    }


                    return `
                        <div class="model-version-card">

                            <div class="model-version-header">

                                <div>
                                    <strong>
                                        ${escapeHTML(
                                            modelType
                                        )}
                                    </strong>

                                    <small>
                                        ${
                                            version !== null
                                                ? `Version v${escapeHTML(version)}`
                                                : "Version —"
                                        }
                                    </small>
                                </div>

                                <div>
                                    ${statusBadge(
                                        status
                                    )}
                                </div>

                            </div>

                            <div class="model-version-metrics">

                                <div>
                                    <span>F1</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                firstDefined(
                                                    model.f1,
                                                    metrics.f1
                                                ),
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>Accuracy</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                firstDefined(
                                                    model.accuracy,
                                                    metrics.accuracy
                                                ),
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>Detection</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                firstDefined(
                                                    model.detection_rate,
                                                    model.detection,
                                                    metrics.detection_rate
                                                ),
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>FAR</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                firstDefined(
                                                    model.false_alarm_rate,
                                                    model.far,
                                                    metrics.false_alarm_rate
                                                ),
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                            </div>

                        </div>
                    `;
                }
            )
            .join("");


    setHTML(
        "modelDecision",
        statusBadge(
            decision || "WAITING"
        )
    );
}


// ============================================================
// EVALUATION
// ============================================================

function renderEvaluation(data) {

    const evaluation =
        data.evaluation || {};

    const comparison =
        Array.isArray(
            evaluation.comparison
        )
            ? evaluation.comparison
            : [];


    const promotion =
        Array.isArray(
            evaluation.promotion
        )
            ? evaluation.promotion
            : [];


    const metadata =
        evaluation.metadata || {};


    const version =
        data.version_management || {};


    const container =
        $("evaluationContainer");


    /*
     * Decision.
     */

    let decision =
        firstDefined(
            metadata.last_decision,
            version.last_decision
        );


    if (
        decision === null &&
        promotion.length
    ) {

        const latestPromotion =
            promotion[
                promotion.length - 1
            ];


        decision =
            firstDefined(
                latestPromotion.decision,
                latestPromotion.status,
                latestPromotion.result
            );
    }


    if (
        decision === null
    ) {

        decision =
            "WAITING";
    }


    setHTML(
        "evaluationDecision",
        statusBadge(
            decision
        )
    );


    /*
     * Evaluation metadata.
     *
     * Prefer score_configuration from dashboard data
     * when available.
     */

    const scoreConfig =
        data.model_metadata &&
        data.model_metadata.score_configuration
            ? data.model_metadata.score_configuration
            : {};


    const evaluationRecords =
        firstDefined(
            scoreConfig.evaluation_records,
            metadata.evaluation_records,
            metadata.records_evaluated,
            comparison.length
        );


    const f1Tolerance =
        firstDefined(
            scoreConfig.f1_tolerance,
            metadata.f1_tolerance
        );


    const farTolerance =
        firstDefined(
            scoreConfig.false_alarm_tolerance,
            metadata.false_alarm_tolerance,
            metadata.far_tolerance
        );


    const fusionThreshold =
        firstDefined(
            scoreConfig.fusion_threshold,
            metadata.fusion_threshold,
            metadata.threshold
        );


    const ifWeight =
        firstDefined(
            scoreConfig.if_weight,
            metadata.if_weight,
            metadata.isolation_forest_weight
        );


    const aeWeight =
        firstDefined(
            scoreConfig.ae_weight,
            metadata.ae_weight,
            metadata.autoencoder_weight
        );


    setText(
        "evaluationRecords",
        integer(
            evaluationRecords
        )
    );


    setText(
        "evaluationF1Tolerance",
        decimal(
            f1Tolerance,
            4
        )
    );


    setText(
        "evaluationFARTolerance",
        decimal(
            farTolerance,
            4
        )
    );


    setText(
        "fusionThreshold",
        decimal(
            fusionThreshold,
            4
        )
    );


    setText(
        "ifWeight",
        decimal(
            ifWeight,
            4
        )
    );


    setText(
        "aeWeight",
        decimal(
            aeWeight,
            4
        )
    );


    /*
     * Evaluation table/card.
     */

    if (!container) {
        return;
    }


    if (!comparison.length) {

        container.innerHTML = `
            <div class="empty-state">
                No model comparison records available.
            </div>
        `;
    }

    else {

        container.innerHTML =
            comparison
                .map(
                    row => {

                        const model =
                            firstDefined(
                                row.model,
                                row.model_name,
                                row.model_type,
                                "MODEL"
                            );


                        const rowVersion =
                            firstDefined(
                                row.version
                            );


                        const f1 =
                            firstDefined(
                                row.f1
                            );


                        const accuracy =
                            firstDefined(
                                row.accuracy
                            );


                        const detection =
                            firstDefined(
                                row.detection_rate,
                                row.detection
                            );


                        const far =
                            firstDefined(
                                row.false_alarm_rate,
                                row.far
                            );


                        return `
                            <div class="evaluation-row">

                                <div>
                                    <strong>
                                        ${escapeHTML(
                                            model
                                        )}
                                    </strong>

                                    <small>
                                        ${
                                            rowVersion !== null
                                                ? `v${escapeHTML(rowVersion)}`
                                                : "—"
                                        }
                                    </small>
                                </div>

                                <div>
                                    <span>F1</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                f1,
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>Accuracy</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                accuracy,
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>Detection</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                detection,
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                                <div>
                                    <span>FAR</span>
                                    <strong>
                                        ${escapeHTML(
                                            decimal(
                                                far,
                                                4
                                            )
                                        )}
                                    </strong>
                                </div>

                            </div>
                        `;
                    }
                )
                .join("");
    }


    /*
     * Decision reason.
     */

    let reason =
        firstDefined(
            metadata.last_decision_reason,
            metadata.reason
        );


    if (
        reason === null
    ) {

        const normalizedDecision =
            normalizeStatus(
                decision
            );


        if (
            normalizedDecision.includes("REJECT")
        ) {

            reason =
                "Candidate did not satisfy the required performance criteria. Production model retained.";
        }

        else if (
            normalizedDecision.includes("PROMOT")
        ) {

            reason =
                "Candidate satisfied the evaluation criteria and was selected for production.";
        }

        else {

            reason =
                "Candidate evaluation completed.";
        }
    }


    setText(
        "evaluationReason",
        reason
    );
}


// ============================================================
// ANOMALIES
// ============================================================

function renderAnomalies(data) {

    const anomalyData =
        data.anomalies;


    let records = [];


    if (
        Array.isArray(
            anomalyData
        )
    ) {

        records =
            anomalyData;
    }

    else if (
        anomalyData &&
        Array.isArray(
            anomalyData.records
        )
    ) {

        records =
            anomalyData.records;
    }


    const tbody =
        $("anomalyTableBody");


    if (!tbody) {
        return;
    }


    const total =
        firstDefined(
            anomalyData &&
                anomalyData.total,
            anomalyData &&
                anomalyData.count,
            data.system &&
                data.system.anomalies_detected,
            records.length
        );


    setText(
        "anomalyTotal",
        integer(total)
    );


    if (!records.length) {

        tbody.innerHTML = `
            <tr>
                <td
                    colspan="7"
                    class="empty-table-cell"
                >
                    No anomalies detected.
                </td>
            </tr>
        `;

        return;
    }


    tbody.innerHTML =
        records
            .slice()
            .reverse()
            .map(
                record => {

                    const features =
                        record.features;


                    let featureText =
                        firstDefined(
                            record.feature_text,
                            record.features_text
                        );


                    if (
                        featureText === null &&
                        Array.isArray(features)
                    ) {

                        featureText =
                            features
                                .map(
                                    value =>
                                        decimal(
                                            value,
                                            3
                                        )
                                )
                                .join(", ");
                    }


                    if (
                        featureText === null &&
                        typeof features ===
                            "object" &&
                        features !== null
                    ) {

                        featureText =
                            Object.entries(
                                features
                            )
                                .map(
                                    ([key, value]) =>
                                        `${key}: ${value}`
                                )
                                .join(", ");
                    }


                    featureText =
                        safe(
                            featureText
                        );


                    return `
                        <tr>

                            <td>
                                ${escapeHTML(
                                    formatTimestamp(
                                        record.timestamp
                                    )
                                )}
                            </td>

                            <td>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            record.if_score,
                                            record.if_anomaly_score,
                                            record.isolation_forest_score
                                        ),
                                        4
                                    )
                                )}
                            </td>

                            <td>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            record.ae_score,
                                            record.autoencoder_score
                                        ),
                                        4
                                    )
                                )}
                            </td>

                            <td>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            record.final_score,
                                            record.fusion_score,
                                            record.fused_score
                                        ),
                                        4
                                    )
                                )}
                            </td>

                            <td>
                                ${escapeHTML(
                                    decimal(
                                        firstDefined(
                                            record.threshold,
                                            record.fusion_threshold
                                        ),
                                        4
                                    )
                                )}
                            </td>

                            <td>
                                ${statusBadge(
                                    firstDefined(
                                        record.status,
                                        "ANOMALY"
                                    )
                                )}
                            </td>

                            <td
                                class="feature-cell"
                                title="${escapeHTML(
                                    featureText
                                )}"
                            >
                                ${escapeHTML(
                                    featureText
                                )}
                            </td>

                        </tr>
                    `;
                }
            )
            .join("");
}


// ============================================================
// SELF CORRECTION
// ============================================================

function renderSelfCorrection(data) {

    const state =
        data.self_correction || {};


    if (
        !state ||
        state.available === false
    ) {
        return;
    }


    const status =
        firstDefined(
            state.status,
            state.state,
            state.current_status
        );


    const message =
        firstDefined(
            state.message,
            state.current_operation,
            status
        );


    if (
        message !== null
    ) {

        setText(
            "pipelineMessage",
            message
        );
    }
}


// ============================================================
// PIPELINE LOG
// ============================================================

function renderPipelineLog(data) {

    const container =
        $("pipelineTerminal");


    if (!container) {
        return;
    }


    const log =
        data.pipeline_log || {};


    let records = [];


    if (
        Array.isArray(log)
    ) {

        records =
            log;
    }

    else if (
        Array.isArray(
            log.records
        )
    ) {

        records =
            log.records;
    }


    if (!records.length) {

        container.innerHTML = `
            <div class="terminal-line">

                <span class="terminal-prefix">
                    SYSTEM
                </span>

                Waiting for pipeline activity...

            </div>
        `;

        return;
    }


    container.innerHTML =
        records
            .slice(-100)
            .map(
                record => {

                    const message =
                        typeof record === "string"
                            ? record
                            : firstDefined(
                                record.message,
                                record.event,
                                record.status,
                                ""
                            );


                    const timestamp =
                        typeof record === "string"
                            ? "SYSTEM"
                            : firstDefined(
                                record.timestamp,
                                record.time,
                                "SYSTEM"
                            );


                    return `
                        <div class="terminal-line">

                            <span class="terminal-prefix">
                                ${escapeHTML(
                                    timestamp
                                )}
                            </span>

                            ${escapeHTML(
                                message
                            )}

                        </div>
                    `;
                }
            )
            .join("");


    container.scrollTop =
        container.scrollHeight;
}


// ============================================================
// COMPLETE DASHBOARD RENDER
// ============================================================

function renderDashboard(data) {

    lastDashboardData =
        data;


    renderSystem(
        data
    );


    renderPipeline(
        data
    );


    renderDrift(
        data
    );


    renderRetraining(
        data
    );


    renderModelMetadata(
        data
    );


    renderTopModel(
        data
    );


    renderModels(
        data
    );


    renderEvaluation(
        data
    );


    renderAnomalies(
        data
    );


    renderSelfCorrection(
        data
    );


    renderPipelineLog(
        data
    );
}


// ============================================================
// ERROR HANDLING
// ============================================================

function showError(error) {

    console.error(
        "Dashboard error:",
        error
    );


    setText(
        "systemStatus",
        "DATA ERROR"
    );


    const dot =
        $("systemDot");


    if (dot) {

        dot.classList.remove(
            "online",
            "warning"
        );


        dot.classList.add(
            "offline"
        );
    }


    /*
     * Do NOT erase already-rendered values.
     * A temporary refresh failure should not destroy
     * the last valid dashboard state.
     */
}


// ============================================================
// LOAD
// ============================================================

async function loadDashboard() {

    try {

        const data =
            await getDashboard();


        renderDashboard(
            data
        );

    }

    catch (error) {

        showError(
            error
        );
    }
}


// ============================================================
// AUTO REFRESH
// ============================================================

function startRefresh() {

    if (
        dashboardRefreshTimer
    ) {

        clearInterval(
            dashboardRefreshTimer
        );
    }


    dashboardRefreshTimer =
        setInterval(
            loadDashboard,
            REFRESH_INTERVAL
        );
}


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadDashboard();

        startRefresh();

    }
);