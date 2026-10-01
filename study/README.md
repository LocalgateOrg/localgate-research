# Study materials

These are verbatim source excerpts from the study application at
[revision `a5c4055c2f2c420d0717282fa5ced7e0f4ea33c2`](https://github.com/LocalgateOrg/Study_LocalModel/tree/a5c4055c2f2c420d0717282fa5ced7e0f4ea33c2).

The excerpts include the blank consent screen, questionnaire items, response
options and instructions. No completed consent forms or participant responses
are included. This directory is a record of the study materials, not a runnable
application; JavaScript interpolation shows the original dynamic form behaviour.
The wording is unchanged. The paper discusses the information and framing given
to participants and the limits of reconstructing what each participant saw.

Copyright and reuse terms are retained in [LICENSE](LICENSE).

[Research guide](../research/README.md) · [Human-study methods](../research/docs/human-study.md)

[Response options](#frequency-response-options) · [Consent](#consent-screen) ·
[Instructions](#background-answer-instructions-and-rating-scale) · [Questionnaire](#post-task-and-demographic-instrument)

## Frequency response options

<details>
<summary>View the original frequency response options (source line 4)</summary>

```javascript
  const FREQ_OPTIONS = ["Never","Less than weekly","1\u20132 days/week","3\u20134 days/week","5\u20136 days/week","Daily"];
```

</details>


## Consent screen

<details>
<summary>View the original consent-screen source (lines 202–274)</summary>

```javascript
  function stepConsent(p, body, foot){
    heading(p, "Before you begin");
    body.innerHTML = `
      <p class="larp-lead">You are invited to take part in a study about how people judge what small AI models can do. Participation is voluntary and you may stop at any time without giving a reason.</p>
      <div class="larp-summary-cols">
        <div><div class="k">What you'll do</div><div class="v">Rate ${state.rows.length || 28} short prompts</div></div>
        <div><div class="k">How long</div><div class="v">About 15 minutes</div></div>
        <div><div class="k">Data collected</div><div class="v">Ratings, timing, demographics</div></div>
      </div>
      <p class="larp-note">Responses are stored without identifying information and reported only in aggregate. You will not be asked to answer the prompts yourself, and there are no right or wrong responses.</p>
      <div class="larp-consent-box">
        <p>You are invited to participate in the online study which investigates knowledge about local model capabilities. The study is conducted by Noah Meissner, Samuel Bullard, Federico Mizzaro and supervised by Dr. David Elsweiler from the University of Regensburg. The study with estimated 30 participants will take place in the period from 2026-08-01 to 2026-08-15. Please note:</p>
        <ul>
          <li>Your participation is entirely voluntary and can be discontinued or withdrawn at any time</li>
          <li>For the evaluation, we collect some basic demographic personal information (e.g., age, gender, etc.)</li>
          <li>The study will last ca. 15 minutes</li>
          <li>You have no direct benefit from participating in the study (unless you receive 0.5 VP hours as a student of the University of Regensburg), but you support our work and help to advance research in this area.</li>
          <li>During the course of the session, all responses entered into the system will be meticulously documented, inclusive of timestamp data.</li>
          <li>Recordings and personal data are treated with confidentiality and will be fully anonymized, stored, evaluated, and potentially published so that no conclusions can be drawn about individual persons anymore</li>
        </ul>
        <p>The option to decline participation is available. For any inquiries, concerns, or complaints regarding the informed consent process or your rights as a research subject, please contact Dr. David Elsweiler. Please read the following information carefully and take the time you need.</p>

        <h3>Purpose and Goal of this Research</h3>
        <p>The purpose of this study is to understand how well people can predict whether a small AI language model that runs locally on everyday devices is able to answer a given question satisfactorily. The goal is to learn whether human judgment is a reliable basis for deciding when a request can be handled by a small local model rather than a larger cloud-based one, which will help inform the design of more efficient and privacy-friendly AI systems.</p>

        <h3>Study Participation</h3>
        <p>Your participation in this online study is entirely voluntary and can be discontinued or withdrawn at any time. You can refuse to answer any questions or continue with the study at any time if you feel uncomfortable in any way. You can discontinue or withdraw your participation at any time without giving a reason. However, we reserve the right to exclude you from the study (e.g., with invalid trials or if continuing the study could have a negative impact on your well-being or the equipment). Repeated participation in the study is not permitted.</p>

        <h3>Study Procedure</h3>
        <ol>
          <li>Participants are initially provided with a brief introduction to the study. After this they will complete the informed consent process.</li>
          <li>This follows by a brief, neutral explanation of local vs. cloud models and of the rating task.</li>
          <li>Participants are assigned to label a stratified sample of open-ended prompts.</li>
          <li>At the end the participants answer demographic questions.</li>
        </ol>
        <p>The confirmation of participation in this study can be obtained directly from the researchers.</p>

        <h3>Risks and Benefits</h3>
        <p>In the online study you will not be exposed to any immediate risk or danger. As with all computer systems on which data is processed, despite security measures, there is a small risk of data leakage and the loss of confidential or personal information. You have no direct benefit from participating in the study (unless you receive 0.5 VP hours as a student of the University of Regensburg), but you support our work and help to advance research in this area.</p>

        <h3>Data Protection and Confidentiality</h3>
        <p>In this study, personal and personally identifiable information is collected for our research. The use of personal or personally identifiable data is subject to the General Data Protection Regulation (GDPR) of the European Union (EU) and will be handled in accordance with the GDPR. This means that you can view, correct, restrict the processing of and have deleted the data collected in this study. Your entries will only be registered in the study with your consent. We plan to publish the results of this and other research studies in scientific articles or other media. Your data will be retained until the study is completed or you contact the researchers to have your data destroyed or deleted. Access to the raw data of the study will be encrypted, password protected during the analysis and only for the authors, colleagues and researchers collaborating on this research. As part of the research work, the data is anonymised using coded identification numbers, whereby no conclusions can be drawn about individual persons without the researchers' information. As no contact details (e.g. emails) are collected, the researchers cannot inform the participants about further details of the study or about a possible breach of confidential data.</p>

        <h3>Identification of Investigators</h3>
        <p><strong>Researchers</strong></p>
        <ul>
          <li>Noah Mei&szlig;ner (<a href="mailto:noah.meissner@stud.uni-regensburg.de">noah.meissner@stud.uni-regensburg.de</a>)</li>
          <li>Samuel Bullard (<a href="mailto:Samuel.Bullard@stud.uni-regensburg.de">Samuel.Bullard@stud.uni-regensburg.de</a>)</li>
          <li>Federico Mizzaro (<a href="mailto:federico.mizzaro@stud.uni-regensburg.de">federico.mizzaro@stud.uni-regensburg.de</a>)</li>
        </ul>
        <p><strong>Principal Investigator</strong></p>
        <p>Dr. David Elsweiler<br>
        <a href="mailto:David.Elsweiler@sprachlit.uni-regensburg.de">David.Elsweiler@sprachlit.uni-regensburg.de</a><br>
        University of Regensburg<br>
        Universit&auml;tsstr. 31<br>
        93053 Regensburg, Germany</p>
      </div>
      <div class="larp-check-row">
        <input type="checkbox" id="larp-consent-cb" ${state.consentChecked?"checked":""}/>
        <label for="larp-consent-cb">I have read the information above and agree to participate.</label>
      </div>
    `;
    p.appendChild(body);
    p.appendChild(foot);
    foot.appendChild(document.createElement("span"));
    const next = navBtn("Continue", () => { state.step = 1; render(); }, {disabled: !state.consentChecked});
    foot.appendChild(next);
    body.querySelector("#larp-consent-cb").onchange = (e) => {
      state.consentChecked = e.target.checked;
      next.disabled = !state.consentChecked;
    };
  }

```

</details>


## Background, answer instructions and rating scale

<details>
<summary>View the original instructions and rating-scale source (lines 331–408)</summary>

```javascript
  function stepBackground(p, body, foot){
    heading(p, "Two places an AI model can run");
    body.innerHTML = `
      <div class="larp-two-box">
        <div class="larp-mode-box local">
          <div class="h"><span class="larp-dot local"></span>On your device</div>
          <div class="d">A smaller model runs locally. Nothing leaves your machine, and it uses less energy.</div>
        </div>
        <div class="larp-mode-box cloud">
          <div class="h"><span class="larp-dot cloud"></span>In the cloud</div>
          <div class="d">A larger model runs on remote servers. Your request is sent over the internet.</div>
        </div>
      </div>
      <p class="larp-lead">Both kinds of model answer the same sorts of questions. Which one suits a given request depends on the request.</p>
      <p class="larp-lead">In this study you will see prompts one at a time and estimate how likely a specific local model, <strong>Gemma4&ndash;e2b</strong>, is to answer each one satisfactorily. You will not answer the prompts, and you will not see the model&rsquo;s answers.</p>
    `;
    p.appendChild(body);
    p.appendChild(foot);
    const back = navBtn("Back", () => { state.step = 1; render(); }, {ghost:true});
    const next = navBtn("Continue", () => { state.step = 3; render(); });
    foot.appendChild(back); foot.appendChild(next);
  }

  // Screen 3 of 4 - what counts as satisfactory
  function stepSatisfactory(p, body, foot){
    heading(p, "What counts as a satisfactory answer");
    body.innerHTML = `
      <p class="larp-lead">Every prompt in this study has a known correct answer. An answer is <strong>satisfactory</strong> if it gives that answer.</p>
      <p class="larp-lead">Think of it as a panel of experts deciding by majority, each with the correct answer in front of them, asked only one thing: does the model&rsquo;s response say the same thing?</p>
      <p class="larp-note">Wording, length, and style are not part of the judgement. Two answers that say the same thing in different words are both satisfactory. A well&ndash;written answer that says something else is not.</p>
      <div class="larp-example">
        <div class="head"><span class="lbl">Correct answer</span><span class="ans">4.0 L</span></div>
        <div class="larp-ex-row"><span class="larp-mark ok">&#10003;</span><span>The volume doubles, so 4.0 L.</span></div>
        <div class="larp-ex-row"><span class="larp-mark ok">&#10003;</span><span>Four litres.</span></div>
        <div class="larp-ex-row"><span class="larp-mark no">&#10007;</span><span>This follows from the gas laws, which relate volume and temperature.</span></div>
      </div>
      <p class="larp-note">You will not see the correct answers during the task, and you are not expected to work them out yourself.</p>
    `;
    p.appendChild(body);
    p.appendChild(foot);
    const back = navBtn("Back", () => { state.step = 2; render(); }, {ghost:true});
    const next = navBtn("Continue", () => { state.step = 4; render(); });
    foot.appendChild(back); foot.appendChild(next);
  }

  // Screen 4 of 4 - the rating scale
  function stepScale(p, body, foot){
    heading(p, "The rating scale");
    body.innerHTML = `
      <p class="larp-lead">Your rating estimates how often the local model would answer a given prompt correctly, if it were asked the same prompt repeatedly.</p>
      <p class="larp-note">The six options divide that proportion into equal ranges covering every possibility, three below an even chance and three above.</p>
    `;
    body.appendChild(buildBandGrid());

    const instinct = document.createElement("div");
    instinct.className = "larp-instinct";
    instinct.textContent = "Answer on instinct. Many prompts are specialist questions from fields you may not know, and you are not expected to work them out. Judging the prompt without knowing the answer is the task.";
    body.appendChild(instinct);

    const expect = document.createElement("div");
    expect.innerHTML = `
      <div class="larp-field">A few things to expect</div>
      <ul class="larp-expect">
        <li>You cannot return to a previous prompt once you&rsquo;ve moved on.</li>
        <li>Every prompt needs a rating &mdash; there is no skip or &ldquo;don&rsquo;t know&rdquo; option.</li>
        <li>You will not see the model&rsquo;s answers or be told whether your ratings were accurate.</li>
        <li>One item asks you to select a specific option instead of giving a rating. It checks that instructions are being read.</li>
      </ul>
    `;
    body.appendChild(expect);

    p.appendChild(body);
    p.appendChild(foot);
    const back = navBtn("Back", () => { state.step = 3; render(); }, {ghost:true});
    const next = navBtn("Start rating", () => { state.idx = 0; state.itemStart = Date.now(); state.step = 5; render(); });
    foot.appendChild(back); foot.appendChild(next);
  }

```

</details>


## Post-task and demographic instrument

<details>
<summary>View the original questionnaire source (lines 474–531)</summary>

```javascript
  function stepPostTask(p, body, foot){
    heading(p, "After the task");
    body.innerHTML = `
      <label class="larp-field">Attention check <span class="larp-field-hint">\u2014 please select \u201cLikely\u201d here.</span></label>
      <select class="larp-input" id="larp-att">
        <option value="">Please select</option>
        ${RATINGS.map(r => `<option value="${r}" ${state.attention===r?"selected":""}>${r}</option>`).join("")}
      </select>
      <label class="larp-field">Optional comments <span class="larp-field-hint">(optional)</span></label>
      <textarea class="larp-input" id="larp-reflect" placeholder="What made your judgements easier or harder?">${state.reflection}</textarea>
    `;
    p.appendChild(body);
    p.appendChild(foot);
    const next = navBtn("Continue", () => { state.step = 7; render(); });
    foot.appendChild(document.createElement("span"));
    foot.appendChild(next);
    body.querySelector("#larp-att").onchange = (e) => state.attention = e.target.value;
    body.querySelector("#larp-reflect").oninput = (e) => state.reflection = e.target.value;
  }

  function stepDemographics(p, body, foot){
    heading(p, "About you", "Only used for statistical analysis, no identifying data.");
    body.innerHTML = `
      <label class="larp-field">Gender</label>
      <select class="larp-input" id="larp-gender">
        <option value="">Please select</option>
        <option value="female">female</option>
        <option value="male">male</option>
        <option value="non-binary">non-binary</option>
        <option value="prefer not to say">prefer not to say</option>
      </select>
      <label class="larp-field">Age</label>
      <input class="larp-input" type="number" id="larp-age" min="14" max="110" />
      <label class="larp-field">Educational background</label>
      <input class="larp-input" type="text" id="larp-edu" placeholder="e.g. Bachelor's, Master's, PhD" />
      <label class="larp-field">Occupation</label>
      <input class="larp-input" type="text" id="larp-occ" placeholder="e.g. student, employee" />
      <label class="larp-field">How often do you use GenAI platforms?</label>
      <select class="larp-input" id="larp-freq2">
        <option value="">Please select</option>
        ${FREQ_OPTIONS.map(f => `<option value="${f}">${f}</option>`).join("")}
      </select>
    `;
    p.appendChild(body);
    p.appendChild(foot);
    const back = navBtn("Back", () => { state.step = 6; render(); }, {ghost:true});
    const next = navBtn("Finish", () => { state.step = 8; render(); });
    foot.appendChild(back); foot.appendChild(next);
    body.querySelector("#larp-gender").onchange = e => state.demo.gender = e.target.value;
    body.querySelector("#larp-age").oninput = e => state.demo.age = e.target.value;
    body.querySelector("#larp-edu").oninput = e => state.demo.education = e.target.value;
    body.querySelector("#larp-occ").oninput = e => state.demo.occupation = e.target.value;
    body.querySelector("#larp-freq2").onchange = e => state.demo.freq2 = e.target.value;
  }

  // ── Submission ──────────────────────────────────────────────────────────
  // POST the finished CSV to the relay (worker/), which stores it and emails it.
  // Retries with backoff on network failure / 5xx / timeout; a 4xx is a bug on our
```

</details>
