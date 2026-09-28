# order_type_and_level_decision

_NotebookLM answer, 2026-09-25 - paraphrase with page citations, UNVERIFIED until checked with tools.nlm.verify_quotes._

## Question

You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system. Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree; give every number the books actually state (percentages, bar counts, probabilities, statistics). Prefer concrete, mechanically testable conditions. The user rejects scaling into multiple simultaneous positions, so do NOT answer with pyramiding/multiple-unit tactics; focus on choosing ONE correct level or ONE correct immediate entry.  QUESTION (CHOOSING THE ORDER TYPE AND THE LEVEL): Synthesize from the books a decision procedure for choosing between (a) a resting limit order at a pullback/reversal level, (b) a buy-stop/sell-stop at a breakout level, and (c) an immediate market entry with the trend. What conditions (trend strength, distance to the nearest support/resistance, volume, pattern type, volatility, time in the move, higher-timeframe alignment) favor each? For (a), how should ONE level be chosen among several candidate supports/resistances (rank the criteria the authors give: first reaction low, prior swing points, retracement zones, moving averages, trendline, gap, round numbers, volume-at-price, confluence of several) and how many independent confirmations at that level are enough? For a reversal (counter-trend) trade at a level versus a trend-continuation trade at a level, what different evidence do the authors require? Include where authors disagree.

## Answer

Here is the complete, deterministic decision procedure for choosing order types, selecting entry levels, and evaluating trade evidence, synthesized strictly from the nine uploaded texts.

---

### 1. Master Decision Procedure: Choosing Order Type & Entry Path

To determine whether to use **(a) a resting limit order at a pullback level**, **(b) a buy-stop/sell-stop at a breakout price**, or **(c) an immediate market entry with the trend**, apply the following seven-variable checklist:

```
                      [EVALUATE 7 VARIABLES]
                                |
      +-------------------------+-------------------------+
      |                         |                         |
[REGIME: RANGING / PULLBACK] [REGIME: BREAKOUT/SETUP] [REGIME: POWER SURGE]
  ADX < 20 / Declining       ADX Crossing > 20          ADX > 40 / Mega OB
  Pullback to S/R            Consolidating at Rim       Cleared Pivot (≤5% Ext)
  Volume Shrinks on Dip      Volume Compression         Volume Surge ≥+50%
      |                         |                         |
      v                         v                         v
(a) RESTING LIMIT ORDER   (b) BUY-STOP / SELL-STOP   (c) IMMEDIATE MARKET ORDER
```

#### Variable 1: Trend Strength & Regime (ADX / Directional Movement)
* **(a) Resting Limit Order:** Favored in **weak/moderate trends or horizontal trading ranges** where ADX is below 20 or declining, and oscillators (RSI, Stochastics) cycle between support and resistance (*Murphy, Ch. 15, pp. 275–278; Schwager, Ch. 10, pp. 148–150*).
* **(b) Buy-Stop / Sell-Stop Order:** Favored when a **new trend is initiating or resuming**, signaled by ADX turning up and crossing above 20 with \\(+DI > -DI\\) (*Murphy, Ch. 15, p. 276; Pring, Ch. 15, pp. 231–237*).
* **(c) Immediate Market Order:** Favored during **extreme trend acceleration**—such as an ADX rising above 40 or a "Mega Overbought" RSI (\\(>70\\)) in a primary uptrend, or a High and Tight Flag where rapid momentum will run without retracing (*Pring, Ch. 13, p. 222; Bulkowski, Ch. 22, p. 356*).

#### Variable 2: Distance to Support / Resistance
* **(a) Resting Limit Order:** Price is currently retracing toward a known underlying support level or broken prior peak (*Nison, Ch. 11, pp. 161–162; Murphy, Ch. 4, pp. 51–53*).
* **(b) Buy-Stop / Sell-Stop Order:** Price is consolidating immediately below resistance/pivot, and the nearest overhead resistance is **\\(>5\%\\) away** (*Bulkowski, Ch. 3, p. 48; O'Neil, Ch. 10, p. 101*).
* **(c) Immediate Market Order:** Price has just cleared the pivot breakout price and is **\\(\le 5\%\\) extended** past the breakout price (*O'Neil, Ch. 10, p. 101, p. 165*).
* **STAY OUT Mandate:** If nearby overhead resistance is **\\(<5\%\\) away** from the breakout price, do **NOT** enter via any order type; nearby overhead supply will retard momentum and degrade the risk/reward ratio (*Bulkowski, Ch. 3, p. 48*).

#### Variable 3: Volume Signature
* **(a) Resting Limit Order:** Volume **contracts/shrinks on the pullback** toward support (*Pring, Ch. 7, pp. 208–212; Murphy, Ch. 7, pp. 160–165*).
* **(b) Buy-Stop / Sell-Stop Order:** Volume dries up inside the consolidation base prior to the break (*Pring, Ch. 8, pp. 217–218*).
* **(c) Immediate Market Order:** Volume on the breakout bar expands **\\(\ge +50\%\\) above its 50-day average** (\\(1.5\times\\) normal volume) (*O'Neil, Ch. 10, p. 165*).

#### Variable 4: Pattern Geometry
* **(a) Resting Limit Order:** Re-testing broken pattern rims (Double Bottom neckline, Rectangle rim, Cup-with-Handle 42% handle retrace) (*Bulkowski, Ch. 9, p. 155; Ch. 14, p. 222; Ch. 38, p. 585*).
* **(b) Buy-Stop / Sell-Stop Order:** High and Tight Flags (0% break-even failure rate; Bulkowski explicitly mandates placing a buy-stop order above the top trendline, *Ch. 22, p. 356*), Cup-with-Handle pivot breakouts, or Ascending Triangles (breaks out upward 77% of the time, *Ch. 47*).
* **(c) Immediate Market Order:** Breakaway gaps on heavy volume out of long flat bases, or secondary 3-day flags formed mid-trend (*Pring, Ch. 9, p. 218; Schwager, Rule 8, p. 290; Ch. 12, p. 164*).

#### Variable 5: Volatility (ATR) & Compression
* **(a) Resting Limit Order:** Volatility is normal/low; protective stop placed \\(1.5\times \text{ATR}\\) below support (*Bulkowski, Ch. 6, p. 112*).
* **(b) & (c) Stop / Market Orders:** A "coiled spring" volatility compression over 3 to 10 days precedes an explosive volatility expansion (*Rockefeller, Ch. 14, p. 209*).

#### Variable 6: Time in the Move & Extension Status
* **(a) Resting Limit Order:** Pullback occurs within **8 to 15 days** of the initial breakout (*Bulkowski, Ch. 13, p. 222; Ch. 21, p. 336; Ch. 37, p. 574*).
* **(b) Buy-Stop / Sell-Stop Order:** Price is directly at the pivot price (\\(0\%\\) extended).
* **(c) Immediate Market Order:** Price is **\\(\le 5\%\\) extended** past the ideal pivot price (*O'Neil, Ch. 10, p. 101*). 
* **DO NOT BUY Rule:** If price is **\\(>5\%\\) past the breakout price**, it is classified as a late entry; do **NOT** chase with a market order (*O'Neil, Ch. 10, p. 101; Schwager, Rule 8, p. 290*).

#### Variable 7: Higher-Timeframe (HTF) Alignment
* All three paths require multi-timeframe alignment across three horizons (e.g., M5 execution aligned with H1/H4/D1 trend direction) (*Rockefeller, Ch. 16, p. 210; Murphy, Ch. 3, p. 45; Pring, Ch. 1, p. 12*).

---

### 2. Selecting ONE Support Level for a Limit Order & Confirmation Counts

When multiple support levels sit below price, the authors provide strict ranking hierarchy and selection rules:

#### A. Ranking Hierarchy for Support Levels
1. **Rank 1: Structural Congestion & Multi-Touch Thick Bases:** Horizontal trading ranges lasting weeks/months, triangle apexes, or multi-touch bases (Triple/Double Bottoms) form "massive support" far stronger than isolated single-bar spikes (*Schwager, Ch. 4, pp. 65–67; Bulkowski, Ch. 18, 50*).
2. **Rank 2: High Volume-at-Price Nodes:** Price levels where the heaviest volume/turnover previously traded (*Murphy, Ch. 4, p. 53; Pring, Ch. 5, p. 178; Bulkowski, Ch. 32, p. 518*).
3. **Rank 3: Change of Polarity (Broken Prior Resistance):** Broken prior reaction peaks or pattern rims (necklines) (*Pring, Ch. 5, p. 175; Murphy, Ch. 4, pp. 51–53; Nison, Ch. 11, pp. 161–162*).
4. **Rank 4: The Combined Retracement Zone (38.2% to 50% / 61.8% Fibonacci):** Standard percentage retracement bands derived from Dow Theory and Fibonacci ratios (*Murphy, Ch. 4, pp. 56–57; Nison, Ch. 12, p. 202*).
5. **Rank 5: Dynamic Support (Rising Moving Averages & Trendlines):** Rising 50-day SMA, 10/20 MAs, or lower channel lines (*Pring, Ch. 5, p. 177; Schwager, Ch. 4, p. 66*).
6. **Rank 6: Gap Boundaries / Candlestick Windows:** Top and bottom boundaries of rising breakaway or continuation windows (*Nison, Ch. 7, pp. 157–158; Pring, Ch. 9, p. 185*).
7. **Rank 7: Round Numbers:** Whole psychological numbers (\$10, \$50, \$100) (*Pring, Ch. 5, p. 176; Murphy, Ch. 4, p. 54*).

#### B. The "First Support" Selection Rule
* **Bulkowski’s First Support Rule:** Empirical data proves declining prices almost always stop at the **FIRST underlying support zone encountered** (within 5% to 10% of the high), even if measure-rules project a deeper drop (*Bulkowski, Ch. 32, p. 518; Ch. 43, p. 667*).
* **Murphy’s Target Adjustment Rule:** When a discrepancy exists between a calculated target and a prominent support level, **adjust your limit price level UP to that first prominent support level** (*Murphy, Ch. 6, p. 219*).

#### C. How Many Confirmations Are Required?
* **Rockefeller’s 3-Indicator Rule:** 1 primary feature (technical level) \\(+ 2\text{nd}\\) independent confirmation (volume/momentum) \\(+ 3\text{rd}\\) confirmation \\(=\\) **EXACTLY 3 confirmations required** for high confidence (*Rockefeller, Ch. 1, p. 192*). Adding a 4th or 5th indicator introduces lagging signals and causes analysis paralysis.
* **Schwager’s 1 Structural Filter Rule:** In quantitative testing, applying **1 solid confirmation condition** (e.g., a 2-day close, a 6-day time delay, or 3 consecutive thrust days) eliminates 6 out of 7 or 7 out of 7 false signals without incurring excessive entry lag (*Schwager, Ch. 14, pp. 242–244*).

---

### 3. Reversal (Counter-Trend) vs. Trend-Continuation Evidence

The authors require fundamentally different technical evidence depending on whether a trade at a level is trend-continuation or counter-trend reversal:

| Technical Element | Trend-Continuation Trade at a Level | Reversal (Counter-Trend) Trade at a Level |
| :--- | :--- | :--- |
| **HTF Trend Context** | HTF trend must be strongly aligned (\\(\text{ADX} > 20\\), rising peaks/troughs) (*Murphy, Ch. 15; Pring, Ch. 1*). | HTF trend must show maturity/exhaustion (\\(\text{ADX}\\) turning down from \\(>40\\)) (*Murphy, Ch. 15, p. 278*). |
| **Volume Signature** | Volume **MUST contract/shrink on the pullback** to the level (*Pring, Ch. 7; Murphy, Ch. 7*). | Requires **Exhaustion / Climax Volume** (parabolic surge) followed by drying volume on retest (*Pring, Ch. 7; Rockefeller, Ch. 6*). |
| **Oscillator Signature** | RSI holds above 40–50 midpoint support (or Stochastics dips \\(<20\\)) and turns back up (*Murphy, Ch. 10; Pring, Ch. 14*). | Must show **Oscillator Divergence** or RSI crossing back inside 70/30 boundaries (*Murphy, Ch. 10; Pring, Ch. 13; Schwager, Ch. 6*). |
| **Price Action Confirmation** | Minimal candle confirmation needed (50% Fib retest limit order allowed) due to trend tailwind (*Schwager, Ch. 8; Nison, Ch. 11*). | **MANDATORY explicit price confirmation required!** Fading an extreme without price confirmation is "a recipe for disaster" (*Schwager, Ch. 10, p. 148; Rule 6, p. 290*). |
| **Required Trigger** | Price bounces off support or completes a minor reaction resumption (*Schwager, Ch. 8*). | Must complete a **Confirmed Chart Reversal** (Double Bottom confirmed past neckline, 97% win rate, *Rockefeller, Ch. 9; Bulkowski, Ch. 14*) OR a **Confirmed Candlestick Reversal** (Hanging Man confirmed by next bar close below body, *Nison, Ch. 4*; Inverted Hammer confirmed by white candle close higher, *Nison, Ch. 5*). |

---

### 4. Author Disagreements & Areas of No Guidance

1. **Limit Pullback Orders vs. Breakout Execution:**
   * **Schwager (Rule 11, p. 290)** strongly favors Market/Stop orders over resting Limit orders to prevent missing runaway trend moves.
   * **Murphy (Ch. 14)** considers waiting for pullbacks a sensible compromise but acknowledges missing major moves.
   * **Bulkowski’s statistics** prove across thousands of patterns that formations *with* throwbacks/pullbacks systematically underperform formations *without* throwbacks (*Bulkowski, Ch. 14, 21, 30, 46*).
2. **Flag Breakout Volume:**
   * **Murphy & Pring** state heavy breakout volume is always required.
   * **Bulkowski (Ch. 21, p. 346, Table 21.7)** presents empirical data showing light breakout volume on regular bull-market flags outperforms heavy breakout volume (26% rise vs. 19%), whereas High and Tight Flags require heavy volume (*Ch. 22*).
3. **Number of Confirmations:**
   * **Rockefeller (Ch. 1, p. 192)** mandates 3 confirming indicators.
   * **Schwager (Ch. 14, pp. 242–244)** proves that 1 structural confirmation condition (e.g., 3 thrust days or time delay) is optimal to avoid entry lag.
4. **Areas of No Guidance:**
   * Sub-minute intraday execution, Level-2 order book depth (DOM queues), VWAP calculations, and algorithmic dark pool execution mechanics are **NOT COVERED** in any of the nine books.

---

I hope this synthesized decision procedure provides a clear, deterministic foundation for your intraday trading system.