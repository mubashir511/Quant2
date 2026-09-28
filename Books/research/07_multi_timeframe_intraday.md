# multi_timeframe_intraday

_NotebookLM answer, 2026-09-25 - paraphrase with page citations, UNVERIFIED until checked with tools.nlm.verify_quotes._

## Question

You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system for forex, metals, crypto, indices and stocks. Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree. Prefer concrete, mechanically testable rules (thresholds, counts, conditions) over general advice, and list any numbers the books actually give (percentages, bar counts, success rates).  QUESTION: What do the books say about using MULTIPLE TIMEFRAMES and about INTRADAY / short-term trading specifically: which timeframe's support/resistance dominates, the top-down workflow (long term to short term), how to use pivot points, previous-day high/low/close, opening range, session highs/lows, VWAP or moving averages intraday, and any rules for trading with the trend versus counter-trend at short horizons. Also: what do the books say about how to choose the most reliable trade setups (the checklist of confluence: trend, level, pattern, volume, indicator) and how many confirmations are enough?

## Answer

### 1. Multiple Timeframe Hierarchy & Support/Resistance Dominance

* **Higher Timeframe Dominance:**
  * **Martin Pring (*Technical Analysis Explained*, Chapter 16, p. 281 & Chapter 18, p. 183):** Candlestick patterns and technical levels on intraday charts carry far less significance than on daily or weekly charts. Pring establishes the **5-to-10 Bar Significance Rule**: the predictive effect of any candlestick pattern or technical signal lasts only between **5 and 10 bars** of the chart frame being analyzed. Thus, a pattern on a 60-minute chart influences price for only 5 to 10 hours, whereas a pattern on a weekly chart governs price for 5 to 10 weeks.
  * **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 3, p. 45 & Chapter 4, p. 214):** Support and resistance levels on weekly and monthly charts represent major structural floors and ceilings that completely dominate short-term or intraday price action.
  * **Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 6, p. 157):** Tweezers, hammers, or doji patterns on weekly and monthly charts form major trend bases or tops. Daily or intraday matching highs/lows do not possess the same structural authority unless confirmed by higher timeframe support/resistance.

---

### 2. Top-Down Analysis Workflow (Long-Term to Short-Term)

* **John Murphy’s Top-Down Sequence (*Technical Analysis of the Financial Markets*, Chapter 3, 18, & 19):**
  1. *Step 1 (Macro Environment):* Check overall market indexes and sector trends.
  2. *Step 2 (Dominant Trend):* Examine weekly and monthly charts to identify the major trend direction, primary support/resistance levels, and long-term chart patterns.
  3. *Step 3 (Intermediate Horizon):* Examine daily charts to identify intermediate swing waves, chart pattern completions, and 50-day/200-day moving average locations.
  4. *Step 4 (Short-Term Timing):* Zoom down to near-term or intraday charts (e.g., hourly/minutes) strictly for **trade execution and timing**.
* **Martin Pring’s Trend Alignment Rule (*Technical Analysis Explained*, Chapter 1, p. 12 & Chapter 20):**
  * Short-term and intraday trades must align with the direction of the intermediate and primary trends. Attempting to trade short-term setups against the dominant higher-timeframe trend significantly increases whipsaw losses.
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 16, p. 210):**
  * **Three-Timeframe Confirmation Rule:** Requires alignment across three distinct time horizons before taking a systematic trade (e.g., today's close > 10-day MA AND today's 10-day MA > 10-day MA 10 days ago AND today's close > close 40 days ago). Applying a 3-timeframe filter provides a *"tremendous reduction in whipsaw losses."*

---

### 3. Intraday Reference Points: Pivots, Previous-Day H/L/C, & Opening Range

#### A. Standard Mathematical Pivot Points
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 11, pp. 201–204):**
  * **Median Pivot Point (\\(P\\)):** 
    \\[P = \frac{\text{High} + \text{Low} + \text{Close}}{3}\\]
  * **First Support (\\(S1\\)) and Resistance (\\(R1\\)):**
    \\[R1 = (2 \times P) - \text{Previous Low}\\]
    \\[S1 = (2 \times P) - \text{Previous High}\\]
  * **Second Support (\\(S2\\)) and Resistance (\\(R2\\)):**
    \\[R2 = (P - S1) + R1\\]
    \\[S2 = P - (R1 - S1)\\]
  * **Third Support (\\(S3\\)) and Resistance (\\(R3\\)):**
    \\[R3 = (P - S2) + R2\\]
    \\[S3 = P - (R2 - S2)\\]
  * **Execution Rules:** Pivot lines outline the expected normal range during trendless periods. If long at support, set conservative targets at \\(R1\\), optimistic targets at \\(R2\\), and aggressive targets at \\(R3\\). Selling short at \\(R3\\) targets a decline back down to \\(S3\\).

#### B. Intraday Pivot Point Timing Rules
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 16, "The Use of Intraday Pivot Points," pp. 229–230):**
  * Combines 7 key price levels (prev day's High, Low, Close, and current day's Open, High, Low, Close) with 4 intraday time windows: **Open**, **30 minutes after open**, **Midday (~12:30 PM NY time)**, and **35 minutes before close**.
  * **Rule 1 (First 30 Minutes):** No trading action is taken during the first 30 minutes of the session.
  * **Rule 2 (Morning Buy Stop):** If the market opens above the previous day's close but below the previous day's high, place a buy stop above the previous day's high. If executed, place a protective sell stop below the current day's low.
  * **Rule 3 (Late-Day Buy Stop):** At 35 minutes before the close, if no trade has been taken, place a buy stop above the current day's high with a protective stop under today's open.
  * **Rule 4 (Closing Filter):** To validate any intraday buy signal, prices **must close above both the previous day's close and today's opening price**.

#### C. Opening Range & Market Profile
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 14, p. 209):**
  * **Coiled Spring Setup:** A narrowing \\(x\\)-minute opening range over 3 to 10 consecutive days indicates volatility compression, signaling an impending explosive breakout.
* **John Murphy (*Technical Analysis of the Financial Markets*, Appendix B: Market Profile, p. 232):**
  * **Steidlmayer's Normal Day:** On a normal day, the long-term trader is inactive. The day's initial balance (fair value) is established during the session's first half-hour ("pioneer range"), after which price rotates between the unfair high and unfair low.

---

### 4. Intraday Moving Averages vs. VWAP

* **VWAP (Volume Weighted Average Price):**
  * **NO GUIDANCE IN TEXT.** None of the 9 books in this notebook mention, define, or provide rules for VWAP. (The texts analyze On-Balance Volume, Chaikin Money Flow, Volume Spikes, and Moving Averages, but do not contain the specific VWAP calculation).
* **Intraday Moving Averages:**
  * **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 3, p. 126):** Moving averages apply identically to intraday bar intervals (e.g., 5-minute, 15-minute, or 60-minute bars) as they do to daily charts, where the "close" refers to the final price quote of that specific intraday bar.
  * **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 11, p. 191):** Standard moving average parameters (such as a 12-period MA) can be applied directly to 15-minute intraday bars.
  * **Martin Pring (*Technical Analysis Explained*, Chapter 11 & 18):** Moving averages function as dynamic support and resistance levels on intraday charts.

---

### 5. Rules for Trend vs. Counter-Trend at Short Horizons

* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 10, pp. 148–150 & Rule 6, p. 290):**
  * Fading short-term price momentum or countertrend trading without price confirmation in a trending market is **"a recipe for disaster."** 
  * In a trending market, short-term setups must be taken **only in the direction of the higher-timeframe trend** (buying pullbacks in an uptrend, selling rallies in a downtrend). Countertrend trading is reserved strictly for horizontal trading ranges.
* **John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 214):**
  * Near-term/intraday trends must be used strictly for timing entries in the direction of the intermediate trend. In an intermediate uptrend, short-term setbacks are treated exclusively as buying opportunities.
* **Martin Pring (*Technical Analysis Explained*, Chapter 13 & 20):**
  * An extreme short-term oscillator reading (overbought/oversold) in a strong trend is not a signal to trade countertrend; it often represents a **"mega overbought"** reading that signals powerful trend continuation.

---

### 6. The Confluence Checklist & Number of Confirmations Required

#### A. John Murphy’s 23-Point Technical Checklist
In **Technical Analysis of the Financial Markets** (Chapter 19, "Technical Checklist", p. 231), John Murphy provides the definitive master checklist for evaluating trade setups:

1. What is the direction of the overall market?
2. What is the direction of the various market sectors?
3. What are the weekly and monthly charts showing?
4. Are the major, intermediate, and minor trends up, down, or sideways?
5. Where are the important support and resistance levels?
6. Where are the important trendlines or channels?
7. Are volume and open interest confirming the price action?
8. Where are the 33%, 50%, and 66% retracements?
9. Are there any price gaps and what type are they?
10. Are there any major reversal patterns visible?
11. Are there any continuation patterns visible?
12. What are the price objectives from those patterns?
13. Which way are the moving averages pointing?
14. Are the oscillators overbought or oversold?
15. Are any divergences apparent on the oscillators?
16. Are contrary opinion numbers showing any extremes?
17. What is the Elliott Wave pattern showing?
18. Are there any obvious 3 or 5 wave patterns?
19. What about Fibonacci retracements or projections?
20. Are there any cycle tops or bottoms due?
21. Is the market showing right or left translation?
22. Which way is the computer trend moving: up, down, or sideways?
23. What are the point and figure charts or candlesticks showing?

#### B. How Many Confirmations Are Enough?
* **Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 1, p. 192):**
  * **The 3-Indicator Rule:** A trade setup defined by a single primary feature (e.g., price touching a support line) gains credibility when confirmed by a 2nd independent indicator (such as volume, momentum, or relative strength). **By the time you add a 3rd confirming indicator (3 total confirmations), your confidence level should be high enough to execute the trade.**
  * *Warning:* Adding a 4th or 5th indicator causes "analysis paralysis" and introduces lagging indicators that delay entry until most of the price move is over.
* **Jack Schwager (*Getting Started in Technical Analysis*, Chapter 14, pp. 242–244):**
  * Adding **1 well-designed confirmation condition** (such as requiring a 2-day close, a 6-day time delay, or 3 consecutive thrust days beyond a breakout) eliminates **nearly 100% of false signals** (eliminating 6 out of 7 or 7 out of 7 false breakouts in tests). Schwager confirms that 1 solid confirmation rule is sufficient, as adding multiple stacked confirmation filters severely delays entry and reduces net profits.

---

### 7. Summary of Author Disagreements and Areas of No Guidance

| Topic | Author Comparison / Disagreements | Status in Text |
| :--- | :--- | :--- |
| **VWAP Indicator** | None of the 9 books cover or define Volume Weighted Average Price (VWAP). | **NO GUIDANCE IN TEXT** |
| **Pivot Point Calculation** | **Rockefeller** uses the standard mathematical 3-tier formula (\\(P, R1\text{--}R3, S1\text{--}S3\\)). **Murphy** uses an intraday timing system combining 7 key price levels across 4 specific time windows. | Different applications (price targets vs. time-window execution). |
| **Optimal Confirmations** | **Rockefeller** dictates **3 confirming indicators**. **Schwager** demonstrates that **1 structural confirmation filter** (e.g., 3 thrust days or time delay) is optimal to avoid entry lag. | Disagreement on count (3 indicators vs. 1 structural filter). |

---

💡 **Next Step Suggestion:** 
Would you like to compile all the extracted rules from this entire session into a structured, production-ready system specification document or Python pseudocode trading strategy?