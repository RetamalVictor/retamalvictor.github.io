---
title: "Practice Project for Systems Engineering - Part 0"
date: "2025-05-22"
tags: ["systems-engineering", "v-model", "rc-racing", "embedded"]
summary: "A prelude to a mini-series applying systems engineering principles to a 1/14-scale RC buggy project. Fresh from an AIAA course on space-grade SE practice, exploring the V-model methodology."
readTime: "5 min"
featured: true
---

# Practice Project for Systems Engineering - Part 0

*A Prelude to the Mini-Series*

The pages that follow are not a build log, at least not yet.  They are, first and foremost, a structured exercise in **applied systems engineering**.  Fresh from an intensive AIAA course on space-grade SE practice, I wanted a sandbox that was small enough to tackle after work yet rich enough to showcase every rung of the ["V" model](https://en.wikipedia.org/wiki/V-model).

A 1/14-scale WLtoys buggy, a $10 ESP32 module, and the perennial headaches of RC racing (run-away cars, video lag) fit that bill perfectly.

<figure style="text-align:center;">
  <!-- direct link to the raw file on Wikimedia's upload server -->
  <img src="/images/learning/Systems_Engineering_Process_II.svg"
       alt="V-model illustrating decomposition on the left and verification on the right"
       style="max-width:100%;height:auto;display:block;margin:0 auto;">
  <figcaption style="font-style:italic;">
    The V-model that underpins this system methodology.
    (Public-domain image, U.S. Government source via Wikimedia Commons.)
  </figcaption>
</figure>
<br>

### What the V-model actually shows
The V-model is a visual shorthand for disciplined, top-down system development followed by bottom-up proof. On the left-hand leg you start with fuzzy stakeholder pains, sharpen them into measurable system requirements, and then decompose those requirements through progressively finer architectural layers until you reach code or hardware drawings.

The bottom point (unit implementation) is where abstraction is lowest and component isolation is highest. The right-hand leg then climbs back upward, but in reverse order: unit tests prove each block, component integration proves every interface, and finally system‐level validation proves that the finished artefact solves the original pain.

Two principles make the diagram more than just a pretty shape:
  1. Bidirectional traceability - every artefact on the left must have a matching verification artefact on the right.
  2. Early test planning - you design test hooks at the same time you design functions, not as an after-thought.

Because of those principles, teams can localise defects, predict schedules, and justify design trade-offs with cold data instead of hope.

**Part 1** captures the *left-hand* descent of the V-model.  It frames the stakeholder pains, distils them into SMART requirements, and surveys the trade space, always with a kitchen-table verification plan in view.  If you are new to SE, **think of Part 1 as the theory section: why clarity of need, aggressive measurability, and early test hooks save months of rework down the line.**

**Part 2** pivots to design: a tri-view architecture (functional, physical, software) that shows, line-by-line, how each requirement is realised and how every design choice earns its keep.  You will see control loops mapped to tasks, latency budgets split across hops, and even a "budget" analogue FPV link added when risk analysis exposed Wi-Fi's soft underbelly.  In short, **Part 2 is the climb up the *right-hand* side of the V, still on paper but already instrumented for proof**.

Will there be a Part 3 with hardware photos and skid-marks on the parking lot?  Not soon, I make no promise.  Another project already competes for bench time. For now, treat this series as a reference walk-through.

---

*© 2025 Victor Retamal - Project Notes.*

**Connect with me:** [![Goodreads](https://img.shields.io/badge/-Goodreads-brown)](https://www.goodreads.com/user/show/72885820-victor-retamal) [![Twitter](https://img.shields.io/badge/-Twitter-blue)](https://twitter.com/Victor_Retamal_) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue)](https://www.linkedin.com/in/victor-retamal/) [![GitHub](https://img.shields.io/badge/-GitHub-gray)](https://github.com/RetamalVictor)