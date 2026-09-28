# pullback_end_level

_NotebookLM answer, 2026-09-25 - paraphrase with page citations, UNVERIFIED until checked with tools.nlm.verify_quotes._

## Question

You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system. Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree; give every number the books actually state (percentages, bar counts, probabilities, statistics). Prefer concrete, mechanically testable conditions. The user rejects scaling into multiple simultaneous positions, so do NOT answer with pyramiding/multiple-unit tactics; focus on choosing ONE correct level or ONE correct immediate entry.  QUESTION (LIMIT / PULLBACK PATH): In an established trend, how do the books tell us WHERE a pullback or correction is most likely to END so that a single resting limit order is placed at the correct reversal level? Cover: percentage retracements (33/50/66 and Fibonacci 38.2/50/61.8, and what share of corrections stop in which zone), retracement depth of flags/pennants/other continuation patterns, throwback/pullback to the breakout price or pattern rim (how often, how far, how long it takes, what predicts it), moving averages and trendlines/channel lines as pullback targets, previous resistance turned support, gap fills, the volume signature of a healthy correction (declining volume) versus a reversal, oscillator readings at the end of a correction in an uptrend/downtrend (RSI, stochastics, MACD zones, divergence), candlestick or price-action signals that a correction has ended, and any statistics on depth. Also: what signs tell you the pullback level you chose is WRONG (correction is really a reversal) and that you should not have a resting order there?

## Answer

### 1. Percentage Retracements & Fibonacci Support Zones

When an asset is in an established trend, the books define specific percentage bands where price corrections are statistically most likely to terminate and reverse:

#### A. Dow Theory & Classic Percentage Zones
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, pp. 56–57 & Chapter 13, p. 333):**
  * **The 33%, 50%, and 66% Rule:** Originating from Dow Theory, price trends in a correction are divided into thirds: a minimum retracement of **33% (1/3)**, a median retracement of **50% (1/2)**, and a maximum retracement of **66% (2/3)**.
  * **Where Most Corrections Stop:** Murphy provides a concrete rule: *"Most corrections will find support in the 38% to 50% retracement zones."* In a strong, powerful trend, the minimum retracement is usually around **38%**. In a weaker trend, the maximum percentage retracement is usually **62%**.
  * **The Combined Retracement Zone:** Murphy combines Dow Theory and Fibonacci parameters to define the primary buying frame:
    * *Minimum Retracement Zone:* **33% to 38%**
    * *Maximum Retracement Zone:* **62% to 66%**
    * *Standard Rounded Filter:* **40% to 60%**
* **Martin Pring (*Technical Analysis Explained*, Chapter 5, p. 176):**
  * Confirms that the **one-third (33.3%)**, **50%**, and **two-thirds (66.7%)** levels represent the primary technical zones to monitor for support on pullbacks during an advance.
* **W. D. Gann Eighths Rule (cited in Murphy, Ch. 4, p. 57):**
  * Gann divides trends into eighths (1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8), but attaches paramount importance to **3/8 (37.5% \\(\approx 38\%\\))**, **4/8 (50%)**, and **5/8 (62.5% \\(\approx 62\%\\))**.

#### B. Fibonacci Retracement Figures
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 12, p. 202):**
  * Identifies **38.2% (rounded to 38%)**, **50%**, and **61.8% (rounded to 62%)** as the premier mathematical retracement ratios derived from the 13th-century Fibonacci sequence. Nison states that the **50% retracement is the most widely monitored level** across market participants.
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 8, p. 131 & Figure 8.3):**
  * Schwager demonstrates placing single entry orders on pullbacks, showing explicit real-world buy signals occurring at the **50%** and **65%** retracement levels (e.g., 3Com Corp and Coffee futures).

---

### 2. Retracement Depth in Continuation & Chart Patterns

#### A. Cup with Handle Retracement Depth
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 9, p. 155 & p. 162):**
  * **Handle Retrace Statistics:** The handle of a cup-with-handle formation is a corrective pullback. Bulkowski’s statistical analysis reveals:
    * The **most common handle retrace is 42%** of the prior cup advance.
    * The second and third most common retrace depths are **35%** and **60%**, respectively (clustering directly around the 38%, 50%, and 62% Fibonacci figures).

#### B. Measured Move Up (MMU) & Down (MMD) Corrective Phase Depth
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 32, p. 509 & Chapter 33, p. 515):**
  * **Corrective Phase Depth:** Prices retrace **between 40% and 60%** of the first leg move in both bull and bear markets.
  * **Averages & Duration:**
    * *Bull Market MMU:* Average corrective retrace is **47%**, lasting an average of **32 days**.
    * *Bear Market MMU:* Average corrective retrace is **50%**, lasting an average of **22 days**.
  * **Impact on Second Leg Length:** The larger the corrective retrace, the higher the probability that the second leg will equal or exceed the first leg. When the retrace is **<38%**, only **22%** have a longer second leg. When the retrace is **between 38% and 62%**, **31%** have a longer second leg. When the retrace is **>62%**, **58%** have a longer second leg.

#### C. Ascending Scallops Depth
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 42, p. 642):**
  * The average retrace from the peak to the handle low compared to the prior advance is **exactly 50%**.

#### D. Flag and Pennant Retracement Time Limits
* **Martin Pring (*Technical Analysis Explained*, Chapter 8, p. 220):**
  * **The 4-Week Maximum Limit:** A flag or pennant in a trend represents a brief consolidation retrace. Pring dictates that a flag taking **more than 4 weeks (28 days)** to develop should be treated with extreme caution, as it loses its high-probability continuation status and signals a deeper trend reversal.

---

### 3. Throwbacks & Pullbacks to Breakout Levels / Pattern Rims

#### A. Historical Frequency (How Often Price Returns)
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*)** provides precise statistical frequencies for how often price returns to the breakout point or pattern rim before resuming the main trend:
  * **Double Bottoms:** **54% to 64%** throwback rate (Chapters 13–16).
  * **Double Tops:** **48% to 61%** pullback rate (Chapters 17–20).
  * **Rectangles:** **53% to 71%** throwback/pullback rate (Chapters 37–38).
  * **Head & Shoulders Tops/Bottoms:** **45% to 64%** throwback/pullback rate (Chapters 24 & 26).
  * **Triangles (Ascending, Descending, Symmetrical):** **37% to 62%** throwback/pullback rate (Chapters 47–49).
  * **Regular Flags:** **43% to 54%** throwback/pullback rate (Chapter 21).
  * **High and Tight Flags:** **54% to 65%** throwback rate (Chapter 22).
  * *Summary Baseline:* Across all classical pattern types, throwbacks and pullbacks occur roughly **50% to 60% of the time**.

#### B. Duration (How Long It Takes to Return to Breakout Level)
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*)** tracks the average time required for price to return to the breakout price:
  * **Rectangles:** **8 to 9 days** (Chapter 37, p. 574; Chapter 38, p. 586).
  * **Double Tops:** **9 to 10 days** (Chapter 17, p. 281).
  * **Double Bottoms:** **9 to 11 days** (Chapter 13, p. 222).
  * **Head & Shoulders:** **10 to 11 days** (Chapters 24 & 26).
  * **Cup with Handle:** **11 days** (Chapter 9, p. 155).
  * **Flags & High/Tight Flags:** **11 to 15 days** (Chapter 21, p. 336; Chapter 22, p. 356).
  * **Pennants:** **12 to 15 days** (Chapter 34, p. 527).

#### C. Predictors of Throwbacks and Pullbacks
Bulkowski identifies two primary structural conditions that predict whether a throwback/pullback will occur:
1. **Nearby Support or Resistance (The 5% Rule):** If an established support or resistance congestion zone is nearby (**less than 5% away** from the breakout price), price will almost certainly be repelled by that zone, triggering a throwback/pullback to the breakout point (*Chapter 21, p. 336; Chapter 30, p. 462*).
2. **Heavy Breakout Volume on Upside Breakouts:** Unusually heavy breakout volume on upside breakouts frequently causes an immediate throwback because *"if everyone buys at the breakout price, who is left to buy in following days?"* (*Chapter 21, p. 346; Chapter 46, p. 706*).

#### D. Performance Impact of Throwbacks / Pullbacks
* **Bulkowski's Critical Finding:** Across almost every chart pattern, **patterns WITH throwbacks/pullbacks significantly underperform patterns WITHOUT throwbacks/pullbacks.**
  * *Example:* Double Bottoms without a throwback average a **43% rise**, compared to only **32%** for those with a throwback (*Chapter 14, p. 222*).

---

### 4. Dynamic Pullback Targets: MAs, Trendlines, Channels, & Role Reversal

#### A. Change of Polarity (Prior Resistance as Support)
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, pp. 51–53):**
  * **Principle:** Once a prior resistance peak is broken on a closing basis, it completely reverses its role and functions as a primary support floor on subsequent pullbacks.
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 11, pp. 161–162):**
  * Refers to this as the **Change of Polarity Principle** and recommends placing limit orders directly at the price level of the broken prior peak.

#### B. Moving Averages & Trendlines as Dynamic Support
* **Martin Pring (*Technical Analysis Explained*, Chapter 5, p. 177 & Chapter 11, p. 186):**
  * **Dynamic Support:** Rising moving averages (e.g., 50-day, 200-day) and up-sloping trendlines function as dynamic support floors during corrections in an uptrend.
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 3, p. 34 & Chapter 4, p. 66):**
  * **Uptrend Channels:** In an uptrend channel, the lower parallel trendline defines the precise low-risk buying target for single entry limit orders.
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 13, p. 164):**
  * Monitors specific moving averages (e.g., 9-period EMA or 20-day SMA) as dynamic support floors during pullbacks.

---

### 5. Gap Fills & Candlestick Windows as Pullback Anchors

* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 7, pp. 157–158):**
  * **Rising Window Support Zone:** A Rising Window (bullish gap) creates a **support zone across its entire span**:
    * *Top of the Window:* The first, immediate support target where price often bounces.
    * *Bottom of the Window:* The ultimate support boundary. If a close occurs below the bottom of the window, the uptrend is negated.
* **Martin Pring (*Technical Analysis Explained*, Chapter 9, p. 218) & John Murphy (*Technical Analysis of the Financial Markets*, Chapter 6, p. 219):**
  * Breakaway and continuation gaps act as major technical support zones where pullbacks typically halt.

---

### 6. Volume Signature: Healthy Correction vs. Reversal

When price pulls back to a support target, volume provides the ultimate diagnostic signature to distinguish a healthy dip from a trend reversal:

* **Martin Pring (*Technical Analysis Explained*, Chapter 7, pp. 208–212 & Chapter 8, p. 218):**
  * **Healthy Correction Signature:** In an established uptrend, a healthy price correction **MUST be accompanied by declining / shrinking volume**. This proves that the price dip is caused merely by a temporary lack of buyers rather than active selling pressure.
  * **Reversal Warning Signature:** If volume **expands or spikes on the pullback / selloff**, it indicates distribution and aggressive selling, signaling that the move is an actual trend reversal rather than a healthy correction.
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 7, pp. 160–165):**
  * **Healthy Dip Rule:** Price Down \\(+\\) Volume Down \\(+\\) Open Interest Down = Normal, healthy liquidation / profit-taking in an uptrend.
  * **Reversal Rule:** Price Down \\(+\\) Volume Up \\(+\\) Open Interest Up = Aggressive new short selling entering the market.

---

### 7. Oscillator Readings at Pullback Termination

Oscillators indicate when an intraday or daily pullback has become overextended within an established trend:

* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 10, pp. 225–235):**
  * **RSI Midpoint / Shifted Support:** In a strong uptrend, the RSI line rarely falls all the way to 30. Instead, it frequently holds above the **50 midpoint level** or finds support in the **40 to 50 zone** during pullbacks. In strong bull markets, the entire overbought/oversold range shifts upward to **80/40**.
  * **Confirmation Trigger:** Entering long when RSI dips into oversold territory (or the 40–50 bull support zone) and then **crosses back above the 30 or 40 line**.
* **Martin Pring (*Technical Analysis Explained*, Chapter 14, pp. 222–225):**
  * In a bull market, the 50 RSI level acts as strong indicator support during price pullbacks.
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 6, pp. 110–117):**
  * Oversold oscillator readings (Stochastics \\(<20\\), RSI \\(<30\\)) during a major higher-timeframe uptrend provide high-probability pullback entry setups.

---

### 8. Candlestick & Price-Action Signals Confirming Pullback End

Instead of placing a blind resting limit order, the authors detail price-action signals that confirm a correction has terminated:

* **Steve Nison (*Japanese Candlestick Charting Techniques*, Part 1 & Part 2):**
  * Reversal candlestick patterns forming at a pullback support level (Fibonacci 38%/50%/62%, broken prior resistance, or moving average):
    * **Hammer / Piercing Pattern / Bullish Engulfing / Morning Star / Southern Doji / Tweezer Bottoms**.
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 8, p. 132 & Appendix A, p. 316):**
  * **Reversal of Minor Reaction Rule:** Define a minor reaction in an uptrend as an \\(N\\)-day relative low (e.g., \\(N = 8\\) or 10 days). Execute entry when price confirms trend resumption by closing above the most recent \\(X\\)-day relative high (e.g., \\(X = 3\\) or 4 days).
  * **Thrust Count Rule:** Count upthrust days during the reaction; enter long immediately when the **Thrust Count reaches 3**.

---

### 9. Invalidation Rules: When a Pullback Level is WRONG

If a resting order is placed at a pullback support target, the authors provide concrete rules for when that level is invalidated and resting orders **MUST BE CANCELED**:

#### A. Schwager’s Consolidation Cancellation Rule
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 17, Rule 19, p. 291):**
  * **Rule 19:** *"If buying into support or selling into resistance and the market consolidates instead of reversing, get out [and cancel resting orders]."* 
  * *Reasoning:* If price arrives at a support target and sits sideways/consolidates on top of the level rather than bouncing sharply, buying demand has vanished. The consolidation indicates a major breakdown is about to occur.

#### B. Nison’s Window & Candle Invalidation Rules
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 7, p. 157):**
  * **Window Invalidation:** If price **closes below the bottom of a rising window (gap)**, the support setup is completely invalidated—close all positions and cancel resting buy orders.
  * **Candle Invalidation:** If price closes below the lower shadow of a hammer or piercing pattern at support, the support level has failed.

#### C. Pring & Murphy Penetration Rules
* **Martin Pring (*Technical Analysis Explained*, Chapter 8, Figure 8.13)** & **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 71):**
  * A closing penetration of a support level by **more than 3%** (or **1%** for short-term/intraday trading), or **2 consecutive daily closes below support**, invalidates the level.
  * If price falls back inside the boundary of the original chart pattern, cancel resting orders immediately.

#### D. Bulkowski’s 5% Busted Rule
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 1, p. 24):**
  * If price falls **more than 5% past the support breakout level**, the pattern is officially "busted" and the trade setup is canceled.

---

### 10. Author Disagreements & Areas of No Guidance

| Topic | Author Comparison / Disagreements | Guidance Status |
| :--- | :--- | :--- |
| **Pullback Limit Orders vs. Breakout Execution** | **Schwager (Rule 11, p. 290)** strongly warns against relying strictly on resting limit orders at pullbacks because the most dynamic, profitable runaway moves never give pullbacks. **Murphy (Ch. 14)** calls waiting for pullbacks a "sensible compromise" but acknowledges missing runaway moves. **Bulkowski’s statistics** prove that patterns with pullbacks systematically underperform patterns without pullbacks. | Disagreement on trade-off: Schwager/Bulkowski favor taking breakouts or market orders over resting limit pullbacks due to performance and missed-move risk. |
| **Exact Retracement Depth** | **Murphy & Pring** emphasize **38%–50%** as the primary zone. **Bulkowski’s data** shows handle retraces average **42%**, measured move corrective phases average **47%–50%**, and scallop handles average **50%**. | General consensus around the 40%–50% band. |
| **Order Book Depth Mechanics** | None of the 9 books cover Level-2 order book market-depth profiles or DOM (Depth of Market) limit-order queue dynamics. | **NO GUIDANCE IN TEXT** (all rules are derived from bar/candlestick charts, volume, and classical TA). |