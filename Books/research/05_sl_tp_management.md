# sl_tp_management

_NotebookLM answer, 2026-09-25 - paraphrase with page citations, UNVERIFIED until checked with tools.nlm.verify_quotes._

## Question

You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system for forex, metals, crypto, indices and stocks. Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree. Prefer concrete, mechanically testable rules (thresholds, counts, conditions) over general advice, and list any numbers the books actually give (percentages, bar counts, success rates).  QUESTION: What do the books say about STOP-LOSS and TAKE-PROFIT management once a trade is open? Cover: initial stop placement (technical stop versus money stop, beyond the swing low/high, percentage or volatility-based, Bulkowski's 'minor low' distances), moving the stop to breakeven, trailing stops (methods, trailing distance, when to start), partial profit taking, measured-move / measure-rule / projection targets, resistance/support-based targets, risk:reward minimums, time stops, when to cancel a resting order or exit a stalled trade, adding to winners (pyramiding) and never averaging down, and re-entry after being stopped out. Give concrete formulas and numbers.

## Answer

### 1. Initial Stop Placement

#### A. Technical Stop versus Money Stop
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 9, pp. 136–140):** 
  * **Technical Stop:** Liquidates a position at or before the price point where price action causes a transition in the technical chart picture [1]. Key technical reference points include:
    1. *Trendlines:* Place sell stops below an uptrend line or buy stops above a downtrend line [2].
    2. *Trading Ranges:* Place sell stops below the lower boundary or midpoint of a trading range [1, 2].
    3. *Flags and Pennants:* Place stops below the lower boundary of a flag or pennant formation [3].
    4. *Wide-Ranging Days:* Place stops below the true low of an unusually wide-ranging day [3].
    5. *Relative Highs and Lows:* Place stops below the most recent relative low (for longs) or above the most recent relative high (for shorts) [4, 5].
  * **Money Stop:** Defined as a protective stop-loss point determined strictly by a dollar-risk level rather than a technical chart level [5, 6]. Schwager recommends using a money stop when the risk implied by even the closest technical level is excessive [5].
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 86):** 
  * Cites the **Turtle 2% Rule** as a primary money stop, where the stop is placed at the price corresponding to a maximum loss of 2% of total starting equity [7].

#### B. Beyond Swing High / Swing Low
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 9, p. 139):**
  * Defines a **relative low** as a bar whose low is lower than the lows of the preceding \\(N\\) days and succeeding \\(N\\) days (where a reasonable range for \\(N\\) is between 5 and 15 days) [5]. For a long position, the protective stop is placed directly below this relative low [4, 5].

#### C. Percentage and Volatility-Based Initial Stops
* **William O’Neil (*How to Make Money in Stocks*, Chapter 9, pp. 91–93):**
  * **The 7% to 8% Percentage Stop:** Establishes a mandatory rule to cut all losses at an absolute maximum of 7% to 8% below the initial purchase price [8-10]. 
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 6, p. 112 & Chapter 18, p. 305):**
  * **1.5x ATR Volatility Stop:** Recommends calculating the average daily trading range (ATR) of the security over the prior month and setting the protective stop no closer than **1.5 times the ATR** below the current low [11, 12].
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 87):**
  * **Average True Range (ATR) Channel Stop:** Uses the average high-low range expanded by a constant (e.g., 25% of the range) [13].
  * **Chandelier Exit (Chuck LeBeau):** Places a trailing stop at a fixed multiple of the Average True Range below the highest high or highest close achieved since entry [13].

#### D. Bulkowski’s "Minor Low / Minor High" Offset Rules
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapters 2, 3, 4, 9, 32, 41):**
  * **The \$0.15 Rule:** Across broadening patterns, cups with handles, flags, and scallops, Bulkowski specifies placing the initial protective stop-loss order exactly **\$0.15 below the nearest minor low** (for long trades) or **\$0.15 above the nearest minor high** (for short trades) [14-19].
  * **The \$0.10 Rule:** For triple bottoms and triple tops, Bulkowski places the stop **\$0.10 below the lowest low** or **\$0.10 above the highest peak** [20-23].

---

### 2. Moving the Stop to Breakeven

* **William O’Neil (*How to Make Money in Stocks*, Chapter 10, p. 107):**
  * **15% Profit Trigger:** When a stock advances 15% or more from its correct entry point, raise the defensive sell line (stop) up to less than 5% below the initial purchase price to eliminate downside risk [8].
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 12, p. 162):**
  * **Breakeven Warning:** Schwager notes a trade-off: moving stops to breakeven too quickly (e.g., within the first 2 weeks) often causes premature liquidation on normal market noise or temporary pullbacks, knocking the trader out of major winning moves [24].
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 16, p. 403):**
  * **Pyramid Trigger:** Requires adjusting protective stops on the entire position to breakeven as soon as a new pyramid unit is added [25].
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 4, p. 95 & Chapter 47, p. 735):**
  * Recommends moving stops to breakeven after a confirmed partial rise/decline or once a security advances by 10% [26, 27].

---

### 3. Trailing Stops

* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 9, pp. 140–142 & Chapter 17, Rule 33):**
  * **Relative Low Trailing:** Trailing stops adjust continuously upward in a bull market by placing the stop below each newly formed relative low (e.g., the lowest low of the past 10 days) [28, 29].
  * **Rule 33:** Recommends using trailing stops supplemented by evolving price action instead of rigid profit targets to avoid cutting major trend profits short [30, 31].
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, pp. 86–87):**
  * **Moving Average Trailing Stop:** Uses a 10-day moving average as a warning to trim positions and a 20-day moving average as the trailing stop line [32].
  * **Parabolic SAR:** Uses SAR dots that accelerate upward toward price during a trend, serving as a trailing exit [13, 32].
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 18, pp. 304–305):**
  * **Progressive Minor High/Low Trailing:** As price creates new minor highs, move the stop up to \$0.15 below the preceding minor low [33-35].

---

### 4. Partial Profit Taking

* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 17, Rules 34 & 38):**
  * **Rule 34 (Rapid Profit Realization):** If 50% to 60% of a price target is realized within 1 week, or 75% to 80% is realized within 2 to 3 weeks, take partial profits immediately and reinstate liquidated lots on a subsequent reaction [30].
  * **Rule 38 (Large Positions):** Avoid wanting to be 100% right; always take partial profits at targets while keeping a partial position for the duration of the trend until a convincing reversal pattern appears [36].
* **William O’Neil (*How to Make Money in Stocks*, Chapter 10, pp. 101–102):**
  * **The 20% Profit Rule:** Take profits when a stock rises 20% to 25% from its breakout point, except for exceptional power leaders [8, 9].

---

### 5. Measured-Move / Measure-Rule / Projection Targets

* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapters 1, 2, 4, 14, 21, 26, 32):**
  * **Standard Formula:** Formation Height = Highest High minus Lowest Low [37]. Target = Breakout Price \\(+\\) Height (for upward breakouts) or Breakout Price \\(-\\) Height (for downward breakouts) [37, 38].
  * **Statistical Reliability:** Full-height measure rules achieve their target 59% to 66% of the time for upward breakouts in bull markets, but only 31% to 54% of the time for downward breakouts [37-39].
  * **The 1/2 Height Conservative Target:** Using **half the formation height** increases the target success rate to **83%** for Measured Move Down patterns and **69%–72%** for Double Tops [40-42].
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 10, pp. 144–146):**
  * Assumes price advances in equal-sized swings (e.g., a 30-cent initial rally projects a 30-cent second leg from the reaction low) [43].
* **Martin Pring (*Technical Analysis Explained*, Chapter 8, pp. 182–184):**
  * Emphasizes that a pattern measuring objective represents a **minimum ultimate objective** rather than an instant target [44].

---

### 6. Support / Resistance-Based Targets

* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 6, p. 219):**
  * **Target Adjustment Rule:** When a discrepancy exists between a pattern's calculated measure target and a clear support/resistance level, **always adjust the price target to that support or resistance level** [45, 46].
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 10, p. 147):**
  * Set exit objectives directly below prior major resistance levels (for long trades) or above major support levels (for short trades) [47].
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 2, p. 37 & Chapter 3, p. 48):**
  * If an established support/resistance zone sits within 5% of your target, adjust your target to just inside that support/resistance area [34, 48, 49].

---

### 7. Risk:Reward Minimums

* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 16, p. 397):**
  * **3-to-1 Minimum Ratio:** Because futures traders win on only ~40% of their trades, a minimum **3:1 reward-to-risk ratio** is required before taking any trade [50].
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 11, p. 179):**
  * **2-to-1 Minimum Ratio:** Recommends taking trades that yield a risk/reward ratio of **1:2 or higher** [51].
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 87):**
  * Recommends requiring a minimum **2:1 ratio** based on realistic worst-case loss vs. worst-case gain estimates [52].

---

### 8. Time Stops

* **William O’Neil (*How to Make Money in Stocks*, Chapter 10, p. 107):**
  * **The 13-Week Selection Test:** If a newly purchased stock has failed to advance after 13 weeks (a full quarter of a year) but has not hit its stop-loss, liquidate the position to free up capital [8, 9].
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 2, Table 2.5):**
  * Statistical data shows almost half (42%–50%) of downward breakouts reach their ultimate low within 2 weeks, whereas upward breakouts in bull markets often take over 70 days [53, 54].
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 14, p. 242):**
  * Systemic time stops liquidate positions if a trade fails to generate a new high/low within \\(N\\) bars after entry [55].

---

### 9. When to Cancel a Resting Order or Exit a Stalled Trade

* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 17, Rules 19 & 43):**
  * **Rule 19 (Order Cancellation):** If buying into support or selling into resistance and price consolidates instead of bouncing, exit immediately and cancel resting orders [30, 56].
  * **Rule 43:** Exit immediately on the first sign of trade failure [36].
* **Martin Pring (*Technical Analysis Explained*, Chapter 8, Figure 8.13):**
  * Exit immediately if price falls back inside a pattern boundary and violates the previous minor low [57].

---

### 10. Adding to Winners (Pyramiding) vs. Never Averaging Down

* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 8, p. 135 & Chapter 17, Rule 12):**
  * **Three Cardinal Rules for Pyramiding:**
    1. Add a unit only if the previous unit shows a profit [58].
    2. Never add a unit if the intended stop implies a net loss on the total combined position [58].
    3. Pyramid units must be **no greater than the base position size** [58].
  * **Rule 12:** Never double up or average down near the original entry point after having been ahead [56].
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 16, p. 403):**
  * **Rule 10 (Pyramiding Guidelines):**
    a. Each successive layer must be smaller than the previous one [25].
    b. Add ONLY to winning positions [25].
    c. **NEVER add to a losing position (never average down)** [25].
    d. Adjust protective stops to breakeven on the entire position [25].
* **William O’Neil (*How to Make Money in Stocks*, Chapter 10, p. 101):**
  * **5% Pyramid Cap:** Never pyramid more than 5% past the pivot buy point (e.g., allocate 60% base position at pivot, 30% at +2–3% advance, and 10% at +4–5% advance) [8, 9].
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 87):**
  * Calls adding to losing positions *"the blackest of cardinal sins in trading"* [59].

---

### 11. Re-Entry After Being Stopped Out

* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 17, Rule 37 & Chapter 14, p. 247):**
  * **Rule 37:** If stopped out of a trade that still possesses long-term potential, maintain a game plan for re-entry [36]. Do not let the fact that the re-entry price is worse than the exit price keep you from getting back in [36].
  * **System Re-Entry Rule:** Any system employing a trade exit rule MUST include explicit re-entry rules; otherwise, the system will miss major trend moves [60].
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 11, p. 278):**
  * **Point & Figure Re-entry:** After being stopped out during a "pole" (a long uninterrupted 10+ box move), re-enter on the first subsequent 3-box reversal in the original trend direction [61].
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 86):**
  * **Rule 5 of Technical Trading:** System design strictly requires establishing an explicit re-entry rule after being stopped out or after hitting a target [62].

---

### 12. Summary of Author Disagreements & Areas of No Guidance

| Subject | Author Comparison / Disagreements | Guidance Status |
| :--- | :--- | :--- |
| **Initial Stop Method** | **O'Neil** uses a strict **7–8% percentage stop** [8, 9]. **Bulkowski** uses **1.5x ATR** or **\$0.15 below minor lows** [11, 12, 14, 15]. **Rockefeller** cites the **2% Turtle money stop** [7]. | Authors propose different stop mechanisms based on asset class (equities vs. futures/forex). |
| **Risk:Reward Minimum** | **Murphy** requires a minimum **3:1** ratio [50]. **Bulkowski & Rockefeller** require a minimum **2:1** ratio [51, 52]. | Disagreement on minimum threshold (3:1 vs 2:1). |
| **Time-based Liquidation** | **O'Neil** liquidates after **13 weeks** of stagnation [8, 9]. **Bulkowski** tracks **14-day** bottom windows for downward breakouts [53, 54]. | Disagreement on holding duration based on investment vs. trading time horizons. |
| **Sub-minute Intraday Execution** | None of the books cover sub-minute HFT order routing, dark pools, or algorithmic market-if-touched execution mechanics. | **NO GUIDANCE in text** (all rules apply to daily/intraday bar charts and standard stop/limit orders). |

---

🎯 **Suggested Next Step:**
Would you like to compile all of these entry, stop-loss, position sizing, and exit rules into a complete, consolidated Python pseudo-code trading algorithm or a markdown system blueprint?