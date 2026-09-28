# support_resistance_ranking

_NotebookLM answer, 2026-09-25 - paraphrase with page citations, UNVERIFIED until checked with tools.nlm.verify_quotes._

## Question

You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system for forex, metals, crypto, indices and stocks. Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree. Prefer concrete, mechanically testable rules (thresholds, counts, conditions) over general advice, and list any numbers the books actually give (percentages, bar counts, success rates).  QUESTION: How do the books identify and rank support and resistance? Cover: zones versus single lines, number of touches, recency, volume at the level, round numbers, gaps, prior swing highs/lows, role reversal (support becoming resistance), moving averages and trendlines as dynamic support, Fibonacci retracements and confluence, and penetration/breakout filters (what counts as a real break). Then answer the practical question: when several support levels lie below price, which one is MOST LIKELY to stop a decline and produce a reversal, and what makes a level stronger or weaker? What do the books say about price stopping at an EARLIER or higher/lower support than the one a trader chose, and how should a trader pick among several candidate supports (e.g. first support versus the strongest support)?

## Answer

Based on the books provided in your notebook, here is how technical analysis authors identify, rank, and trade support and resistance (S/R) levels.

---

### 1. Identification: Zones vs. Single Lines

* **Support/Resistance as Price Zones (Narrow Bands):**
  * **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 4, pp. 67–70):** Explains that support and resistance are rarely single exact price points. Instead, relative highs and relative lows tend to concentrate in relatively narrow bands, forming **"support zones"** and **"resistance zones."** He also introduces **Price Envelope Bands** (e.g., a 20-day moving average \\(\pm\\) 2.5% or 3%) to define upper and lower dynamic boundaries encompassing price action.
  * **Martin Pring (*Technical Analysis Explained*, Chapter 5, pp. 174–178):** Defines support as *"buying (actual or potential) sufficient in volume to halt a downtrend"* and resistance as *"selling sufficient in volume to satisfy all bids."* He stresses that these represent **concentrations of demand and supply** across a price range (a floor or ceiling) rather than a rigid single mathematical line.
  * **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 7, p. 157):** Highlights that price voids or **windows (gaps)** create support/resistance *zones*. For instance, a large rising window between \$20.50 and \$22.50 creates a **\$2.00 price zone of support** from the top of the window down to the bottom.
  * **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 50):** Refers to support and resistance as *"a level or area on the chart under the market where buying interest is sufficiently strong."*
* **Single Price Lines:**
  * **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 196):** Notes that while many technicians draw exact single lines along the daily lows or closes, traders must examine their specific market's habits to determine whether the low-of-the-day or the close defining the line is what the market respects.

---

### 2. Determinants & Ranking of Support/Resistance Strength

The books outline several quantifiable criteria to evaluate whether one support level is stronger or weaker than another:

#### A. Volume & Turnover at the Level
* **Rule:** The greater the volume traded at a specific price level, the more significant that level becomes as support or resistance.
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 53):** States that if a support level is formed on **heavy volume**, it indicates a large number of units changed hands, marking it as significantly more important than a level formed on light volume because more market participants have a vested psychological interest in that price.
* **Martin Pring (*Technical Analysis Explained*, Chapter 5, p. 178):** *"The Amount of a Security that Changed Hands in a Specific Area—the Greater the Activity, the More Significant the Zone."*
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 32, p. 518):** *"A prior peak with high turnover will give more support to a stock on its way down. That is not to say that the stock will not burn through the support, just that it might take more of a push to fall off the cliff."*

#### B. Number of Touches / Tests
* **Martin Pring (*Technical Analysis Explained*, Chapter 6, p. 181):** A line or price level derives its authority from the number of times it has been touched or approached: *"the larger the number, the greater the significance."* An approach (a close encounter) is nearly as important as an actual touch.
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 196):** *"The more times that a low-of-the-day touches the support line without crossing it, the more confidence you should have that it is a valid description of the trend."*
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapters 4, 37, 52):** Gives explicit test/touch counts for structural pattern boundaries:
  * **Rectangles (Chapter 37, p. 584):** Requires a minimum of **2 distinct touches** on both top and bottom trendlines (4 total touches).
  * **Broadening Tops (Chapter 4, p. 18):** Requires at least **3 distinct touches** on each trendline.
  * **Wedges (Chapter 52, p. 797):** Requires a minimum of **5 touches** across the two converging trendlines to safeguard against misidentification.

#### C. Recency
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 53):** Recency directly impacts level strength: *"The more recent the trading took place, the more significant the support or resistance level becomes"* because market participants still remember their positions and breakeven expectations clearly.
* **Martin Pring (*Technical Analysis Explained*, Chapter 5, p. 178):** Notes that the significance of a zone decays as the elapsed time since the zone was last encountered increases.

#### D. Prior Swing Highs/Lows & Structural Thickness
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 4, pp. 65–67):** Distinguishes single absolute peaks/nadirs from **concentrations of relative highs and lows**. Horizontal consolidation regions (sideways trading ranges lasting weeks or months) create far more potent support zones than isolated 1-bar spikes.
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 18, 19, 50):** Demonstrates that horizontal price congestion blocks, triangle apexes (e.g., a symmetrical triangle apex), and multi-touch bases (like Triple Bottoms) create *"massive support"* that frequently halts downward moves entirely.

#### E. Role Reversal ("Change of Polarity")
* **Rule:** Once a support level is decisively broken, it reverses its role and becomes resistance on subsequent rallies. Conversely, broken resistance becomes support on subsequent pullbacks.
* **Martin Pring (*Technical Analysis Explained*, Chapter 5, p. 175):** *"Major Technical Principle: Support reverses its role to resistance on the way up."* Pring attributes this to elementary psychology: trapped buyers who suffered losses sell when price returns to their breakeven level.
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, pp. 51–53):** Explains that all previous selling at a resistance peak becomes buying interest underneath the market once the peak is decisively penetrated.
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 11, pp. 161–162):** Refers to this as the **Change of Polarity Principle** and uses it extensively to identify high-probability entry zones.
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 4, p. 66):** Confirms that following a sustained penetration of a prior high, that prior high becomes support.

#### F. Round Numbers
* **Martin Pring (*Technical Analysis Explained*, Chapter 5, p. 176):** Notes that support and resistance naturally form at round psychological numbers (e.g., 10, 50, 100, 1,000) because traders base decisions on easy psychological landmarks.
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 54):** Highlights the tendency of markets to respect whole numbers (\$10, \$20, \$50, \$100). He provides the concrete rule: **Place protective stops below round numbers for long positions and above round numbers for short positions.**

#### G. Gaps / Windows
* **Martin Pring (*Technical Analysis Explained*, Chapter 8, p. 185):** *"The upper and lower areas of gaps represent potentially significant support and resistance areas."*
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 7, pp. 157–158):** *"The entire rising window becomes a potential support zone."* In a falling window, the entire gap acts as overhead resistance.
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 6, p. 219):** Prominent gaps beneath current prices function as strong support areas.

#### H. Dynamic Support: Moving Averages & Trendlines
* **Martin Pring (*Technical Analysis Explained*, Chapter 5, p. 177 & Chapter 11, p. 186):** *"Major Technical Principle: Moving averages should be thought of as a dynamic level of support and resistance."* Up trendlines and rising MAs act as dynamic support; down trendlines and falling MAs act as dynamic resistance.
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 13, p. 164):** Recommends monitoring specific moving averages (e.g., a 9-period EMA or 20-day MA) that have been successfully defended in the past as dynamic support. **Caveat:** Nison explicitly states *never* to trade solely because price touches a moving average; wait for a confirming candlestick signal (like a hammer or bullish engulfing pattern) at the MA.

#### I. Percentage Retracements & Fibonacci Confluence
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 57):** Standard percentage retracement zones include **33% (1/3)**, **50% (1/2)**, and **66% (2/3)**, along with Fibonacci retracements at **38% (38.2%)** and **62% (61.8%)**. He notes that most market corrections find support within the **38% to 50% retracement zone**.
* **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 12, pp. 162–163):** Demonstrates **"Convergence" (Confluence)**—when a 50% or 38.2% Fibonacci retracement aligns with a broken prior peak (Change of Polarity) and a bullish candlestick pattern, the combined level represents an exceptionally strong support zone.
* **Martin Pring (*Technical Analysis Explained*, Chapter 18, p. 190):** States that Fibonacci levels should *never* be used in isolation, but only in confluence with trendlines, chart patterns, and momentum indicators.

---

### 3. Penetration & Breakout Filters (What Counts as a Real Break)

When testing whether an S/R level or trendline is truly broken or merely experiencing a temporary penetration ("whipsaw"), the authors define the following testable filters:

* **Closing Price Filter:**
  * **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 71):** A daily close beyond an S/R level or trendline is significantly more valid than an intraday penetration.
* **Percentage Price Filters:**
  * **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, pp. 71–72 & Chapter 6, p. 220):**
    * **3% Price Filter:** For major, long-term support/resistance levels or major trendlines, require price to close at least **3% beyond** the level. (e.g., if support is broken at \$400, price must close at or below \$388).
    * **1% Price Filter:** Used for shorter-term or intraday trading where a 3% filter would give away too much move.
  * **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 14, p. 241):** Employs a **2% closing penetration filter** to eliminate bad signals in breakout systems.
  * **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, pp. 200–202):** Notes that while a 3% filter was standard in classic literature, traders often test percentage filters from **10% to 20%** for volatile equities, depending on backtested asset behavior.
* **Time Filters:**
  * **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 71):** **The Two-Day Rule:** Price must close beyond the support or resistance level for **2 consecutive days**. A 1-day violation does not count.
  * **Friday Close Filter (Murphy, Ch. 4):** Requiring a Friday close beyond the level to confirm a weekly signal.
  * **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 14, p. 241):** **Time Delay Filter:** Requires price to remain closed beyond the breakout level for **6 or more days** to validate the signal.

---

### 4. Practical Question: Choosing Among Multiple Support Levels

When several support levels lie below the current price, the books provide specific guidelines on which level is most likely to hold and how a trader should choose among them:

#### A. Which Level is MOST LIKELY to Stop a Decline?
A support level's likelihood of stopping a decline is determined by its **ranking score (Confluence & Density)**:
1. **The Thickest / Most Congested Level:** A multi-month horizontal consolidation zone or triangle base will stop a decline far more reliably than an isolated 1-day swing low (**Schwager**, Ch. 4; **Bulkowski**, Ch. 18, 50).
2. **High Volume Nodes:** Levels where the highest volume previously traded are primary candidates for stopping a move (**Murphy**, Ch. 4; **Pring**, Ch. 5; **Bulkowski**, Ch. 32).
3. **Confluence Zones:** A level where multiple independent technical factors coincide (e.g., a 50% Fibonacci retracement + a rising 50-day moving average + a round number + a change-of-polarity prior peak) has the highest probability of producing a reversal (**Nison**, Part 2; **Pring**, Ch. 5).

#### B. What do the Books Say About Price Stopping at an EARLIER (Higher) Support?
* **Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 32, p. 518; Chapter 43, p. 667; Chapter 51, p. 785):**
  * Bulkowski's empirical data shows that declining prices frequently stop at the **first underlying support zone** encountered, even if mathematical targets (like measure rules) predict a deeper decline.
  * He highlights that if a stock has two support levels below it (e.g., Support 1 at \$40 and Support 2 at \$39, with a target at \$38.25), price will often stall, consolidate, or completely reverse at **Support 1 (\$40)**.
  * **Bulkowski's Mandatory Check:** *"Before taking a position... look for underlying support. If underlying support is too close to the breakout or too robust, then search for another trade... Prices usually have enough power to push through nearby support (those less than 5% away),"* but robust horizontal support within 5% to 10% will frequently kill the trade early.

#### C. Adjusting Targets and Choosing Candidate Supports
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 6, p. 219):**
  * **The Target Adjustment Rule:** *"As a general rule, when a slight discrepancy exists between a projected price target and a clearcut support or resistance level, it’s usually safe to adjust the price target to that support or resistance level."*
  * If a pattern projects a downside target of \$30, but a prominent, high-volume support zone sits at \$32, **you must adjust your price target up to \$32** rather than hoping for \$30.
* **Entry & Stop-Loss Placement Strategy:**
  * **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 14, p. 398):** Place initial protective stops **just below the nearest valid support point** rather than a distant one.
  * **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 13, p. 164 & Chapter 16):** Do **not** place a limit order blindly at the strongest or deepest support level. Instead, watch price as it approaches the **first candidate support level** and wait for market reaction. If a bullish candlestick reversal signal (hammer, piercing line, bullish engulfing) forms at the *first* support level, trade that level. If no candlestick reversal forms and price breaches the first level, wait for the *next* confluence support zone.

---

### Summary Table for Deterministic System Design

| Feature | Primary Filter / Condition | Source / Author |
| :--- | :--- | :--- |
| **Zone Width** | Relative high/low concentration band or Price Envelopes (\\(\pm 2.5\%\\)) | Schwager (Ch. 4) |
| **Window Support Zone** | Span between top of gap and bottom of gap | Nison (Ch. 7) |
| **Valid Trendline / Rectangle** | Minimum 2 touches (Rectangles), 3 (Broadening), 5 (Wedges) | Bulkowski (Ch. 4, 37, 52) |
| **Major Breakout Price Filter** | 3% close beyond level (1% for short-term/intraday) | Murphy (Ch. 4) |
| **Breakout Time Filter** | 2 consecutive daily closes beyond level | Murphy (Ch. 4) |
| **Round Number Stops** | Long stops *below* whole numbers; short stops *above* | Murphy (Ch. 4) |
| **Target Adjustment Rule** | Truncate measure-rule targets to the *first* prominent S/R zone | Murphy (Ch. 6), Bulkowski (Ch. 43) |
| **Execution Trigger at S/R** | Require a confirming Eastern candlestick pattern before entry | Nison (Ch. 13) |
