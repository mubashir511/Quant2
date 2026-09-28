# entry_alternatives

_NotebookLM answer, 2026-09-25 - paraphrase with page citations, UNVERIFIED until checked with tools.nlm.verify_quotes._

## Question

You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system for forex, metals, crypto, indices and stocks. Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree. Prefer concrete, mechanically testable rules (thresholds, counts, conditions) over general advice, and list any numbers the books actually give (percentages, bar counts, success rates).  QUESTION: Limit orders at a pullback often never trigger because the market runs away. What alternatives do the books give for ENTERING a trade: buy-stop/sell-stop above resistance, market-if-touched, entering on a close beyond a level, entering on confirmation of a reversal (candle or indicator), scaling in with multiple units, entering on a minor reaction resumption, waiting for pullback versus taking the breakout, how far past a breakout is too late, and what they say about handling a MISSED trade (re-entry rules, chasing, time limits for a resting order). Give concrete conditions and numbers, and each author's stated trade-offs.

## Answer

### 1. The Core Entry Dilemma: Waiting for Pullbacks vs. Taking the Breakout

#### John Murphy (*Technical Analysis of the Financial Markets*, Chapter 14: "Tactics on Breakouts: Anticipation or Reaction?")
Murphy explicitly outlines the three entry choices every trader faces when a breakout occurs, detailing the exact trade-offs for each:
1. **Anticipating the Breakout (Buying inside the pattern before the break):**
   * *Trade-off:* Yields a better (lower for longs, higher for shorts) entry price if the breakout occurs. However, the probability of a bad trade or false signal is highest.
2. **Taking the Breakout Itself (Buying as the level is breached):**
   * *Trade-off:* Increases the probability of success compared to anticipating, but penalizes the trader with a worse (higher/lower) entry price and slippage.
3. **Waiting for the Pullback/Reaction (Buying on the retest):**
   * *Trade-off:* Represents a "sensible compromise" providing a better entry price and lower risk, *provided the pullback occurs*. However, Murphy warns that the most dynamic, powerful, and profitable markets frequently run away without giving the patient trader a second chance. The primary risk of waiting for a pullback is missing the move entirely.
* **Murphy's Multi-Unit Solution:** If trading multiple units, Murphy recommends splitting the entry: enter 1 unit in anticipation, 1 unit on the actual breakout, and 1 unit on the pullback.

#### Jack Schwager (*Getting Started in Technical Analysis*, Chapter 9, p. 138 & Chapter 11, p. 164)
Schwager confirms that waiting for a retracement to the opposite side of a breakout (e.g., the boundary of a flag or trading range) is a common entry tactic:
* *Quote/Paraphrase:* *"Retracements to this area are so common that many traders prefer to wait for such a reaction before initiating a position... In many instances it will provide better fills, but it will also cause the trader to miss some major moves."* (Chapter 9, p. 138).

#### Thomas Bulkowski (*Encyclopedia of Chart Patterns*)
Bulkowski quantifies the exact statistical probability of throwbacks (upward breakouts) and pullbacks (downward breakouts) across hundreds of pattern samples:
* **Frequency of Retracements:**
  * **Regular Flags:** 41% to 56% throwback/pullback rate (Chapter 21, p. 336).
  * **High and Tight Flags:** 54% to 65% throwback rate (Chapter 22, p. 356).
  * **Rectangles:** 64% to 71% throwback/pullback rate (Chapter 38, p. 585).
  * **Double Bottoms / Double Tops:** 54% to 64% throwback/pullback rate (Chapters 14–17).
  * **Head and Shoulders:** 45% to 64% pullback rate (Chapter 26, p. 408).
* **Bulkowski’s Critical Finding on Performance:**
  * Across almost every chart pattern, **patterns WITH throwbacks/pullbacks significantly underperform patterns WITHOUT throwbacks/pullbacks.**
  * *Examples:* 
    * Double Bottoms without a throwback average a **43% rise**, compared to only **32%** for those with a throwback (Chapter 14, p. 222).
    * Island Reversals without a throwback rise **38%**, compared to **18%** with a throwback (Chapter 30, p. 462).
  * *System Implication:* Waiting for a pullback means systematically filtering for trades that statistical data proves will yield smaller post-breakout gains on average.

---

### 2. Alternative Entry Method 1: Buy-Stop / Sell-Stop Orders Above Resistance / Below Support

#### Jack Schwager (*Getting Started in Technical Analysis*, Chapter 4, p. 58 & Chapter 14, p. 240)
* **Mechanics:** Placing a buy-stop order above resistance (or sell-stop below support) ensures automatic execution the instant price touches or penetrates the level.
* **Trade-offs & Warnings:** Stop orders placed just beyond obvious support/resistance levels are prone to "stop runs" and false breakouts because stop-loss clusters sit in those exact regions. Once the initial flurry of stops is filled, price often falls back into the range unless real buying/selling volume follows (Chapter 4, p. 58).

#### Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 22, p. 356)
* **High and Tight Flag Buy-Stop Rule:** For high and tight flags (which average a 69% gain with a 0% break-even failure rate), Bulkowski explicitly advises using a buy-stop order:
  * *Rule:* *"Since the price rise in the first week is the largest one-week change you are likely to see, place a stop order to buy the stock just above the HTF trend-line boundary."*

---

### 3. Alternative Entry Method 2: Market Orders vs. Limit Orders

#### Jack Schwager (*Getting Started in Technical Analysis*, Rule 11, p. 290 & Chapter 12, p. 162)
* **Rule 11 (Market vs. Limit Orders):**
  * *Exact Wording:* *"In most cases, use market orders rather than limit orders. This is especially important when liquidating a losing position or entering a perceived major trading opportunity—situations in which traders are apt to be greatly concerned about the market getting away from them. Although limit orders will provide slightly better fills for a large majority of trades, this benefit will usually be more than offset by the substantially poorer fills, or missed profit potential, in those cases in which the initial limit order is not filled."* (p. 290).
* **Converting Limit Orders to Market:** Schwager notes in real-world trade logs that if a limit order is not filled and the market begins moving strongly in the intended direction, the trader should immediately convert the limit order to a **market order** rather than abandoning the trade idea (Chapter 12, p. 162).

#### John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, p. 54)
* **The Round-Number Execution Rule for Limit Orders:**
  * *Rule:* Never place limit entry orders directly on obvious round numbers (\$10, \$50, \$100, \$1,000). Place buy-limit orders **just above** round numbers (e.g., \$40.10 instead of \$40.00) and sell-limit orders **just below** round numbers (e.g., \$49.90). Because the crowd clusters limit orders at round numbers, price often touches or approaches the round number without filling orders at the level.

---

### 4. Alternative Entry Method 3: Entering on a Close Beyond a Level (Closing Filters)

#### Jack Schwager (*Getting Started in Technical Analysis*, Chapter 14, pp. 240–244)
* **Daily Close Penetration Filter:** Requires the session to **close beyond** the breakout level before entering, filtering out intraday spikes.
* **Time Delay Filter (6-Day Rule):**
  * *Rule:* Require the market to close beyond the signal price at any time **6 or more days beyond** the original signal date before executing the trade.
  * *Test Result:* Schwager demonstrates that applying this 6-day time delay confirmation condition to a basic 12-day breakout system **eliminated 6 out of 7 false signals** (Figure 14.10, p. 244).

#### John Murphy (*Technical Analysis of the Financial Markets*, Chapter 4, pp. 71–72)
* **Two-Day Rule:** Require price to close beyond the support/resistance or trendline level for **2 consecutive days**.
* **1% / 3% Price Penetration Filter:** Require price to close at least **3% beyond** a major long-term level (or **1% beyond** for short-term/intraday trading).

---

### 5. Alternative Entry Method 4: Confirmation of Reversal (Candlestick & Indicator Setups)

#### Jack Schwager (*Getting Started in Technical Analysis*, Chapter 7, pp. 122–123 & Rule 6, p. 290)
* **3-Day Post-Signal Confirmation Filter:**
  * *Rule:* After an indicator or chart pattern generates a signal, wait 3 days. For a buy signal, enter the trade **only if the close is above the highest high reached since the signal was received** (or on the first subsequent day fulfilling this condition).
  * *Result:* This filter eliminated losing March and May signals in Eurodollar tests while preserving the winning breakout (p. 123).
* **Rule 6 (Pattern Confirmation over Fading):**
  * *Rule:* *"When looking for a major reversal in a trend, it is usually wiser to wait for some pattern that suggests that the timing is right rather than fading the trend at projected objectives and support/resistance points... prices will normally pull back to test highs and lows—often a number of times."* (p. 290).

#### Steve Nison (*Japanese Candlestick Charting Techniques*, Chapter 4, p. 34 & Chapter 12)
* **Candlestick Confirmation Entry:** Never enter blindly at a support level. Wait for a confirming candlestick reversal pattern:
  * **Hanging Man Entry:** Mandatory confirmation required—enter short only on the session *after* the hanging man closes below the hanging man’s real body (Chapter 4, p. 34).
  * **Inverted Hammer Entry:** Mandatory confirmation required—enter long only if the next candle is a white candle closing higher (Chapter 5, p. 78).

---

### 6. Alternative Entry Method 5: Scaling In and Pyramiding (Multiple Units)

#### Jack Schwager (*Getting Started in Technical Analysis*, Chapter 8, p. 135 & Chapter 14, p. 246)
Schwager equates midtrend entries to pyramiding and establishes **Three Cardinal Rules for Pyramiding**:
1. **Profit Prerequisite:** Never add a new unit to an existing position unless the last unit placed is showing a profit.
2. **Total Risk Cap:** Never add a unit if the intended stop point would imply a net loss for the entire combined position.
3. **Unit Sizing:** Successive pyramid units must be **no greater than the base (initial) position size** (e.g., base unit = 2 contracts, pyramid 1 = 2 contracts, pyramid 2 = 1 contract).

* **Schwager's Mechanized Pyramiding System Rules (Chapter 14, p. 246):**
  * *Step 1:* A reaction is defined when net position is long and price closes below the prior **10-day low**.
  * *Step 2:* Once a reaction is defined, initiate an additional long unit on any subsequent **10-day high** if:
    a. The pyramid signal price is above the price of the most recent long entry.
    b. Total position size is less than 3 units (capping the system to a maximum of **2 pyramid units**).
  * *Pyramid Stop Rule:* Liquidate all pyramid positions if an opposite signal occurs OR if the market closes below the low established since the most recent reaction (p. 246).

#### Barbara Rockefeller (*Technical Analysis For Dummies*, Chapter 5, p. 90 & Chapter 16, pp. 290–291)
* **Margin Profit Scaling Rule:** Add a second unit to a 50% margin position only when paper profits are large enough to fully cover the minimum initial margin requirement of the new position.
* **Pyramiding Warning:** Pyramiding without strict protective stops on all units is one of the most common causes of total trader ruin (p. 291).

#### William O’Neil (*How to Make Money in Stocks*, Chapter 10)
* **Pyramid Allocation:** Buy the initial base position (e.g., 60% of total intended capital) on the exact pivot breakout. Add a smaller secondary unit (30%) when the stock advances 2% to 3% above the breakout, and a final small unit (10%) on a minor pullback, stopping all purchases once the stock is 5% past the pivot.

---

### 7. Alternative Entry Method 6: Reversal of Minor Reaction / Resumption Rules

#### Jack Schwager (*Getting Started in Technical Analysis*, Chapter 8, p. 134 & Appendix A, p. 316)
* **Mechanized Midtrend Entry Rules (N-Day Low / X-Day High):**
  * *Uptrend Entry Condition:* Wait for a minor reaction defined as an **N-day relative low** (a new low lower than the lowest low of the prior N days, e.g., N = 8 or 10 days). Enter long when price resumes the trend by closing above the most recent **X-day relative high** (e.g., X = 3 or 4 days).
* **Schwager's Thrust Count Method (Appendix A, p. 316):**
  1. Define a reaction in a rising market (e.g., a close below the 10-day low). Set the *Thrust Count* to 0.
  2. Increment the Thrust Count by 1 on each **upthrust day** (a day where the close is above the prior day's high).
  3. Reset the Thrust Count to 0 anytime the reaction low is penetrated.
  4. **Execute Long Entry:** Enter the trade immediately when the **Thrust Count reaches 3**. Place the protective stop below the reaction low.

#### Martin Pring (*Technical Analysis Explained*, Chapter 8, Figure 8.14)
* **Retracement Trendline Breakout Entry:** Draw a short-term down-sloping trendline across the high points of the pullback move. Execute the long entry the moment price breaks out above that short-term retracement trendline.

---

### 8. How Far Past a Breakout is "Too Late"?

#### William O’Neil (*How to Make Money in Stocks*, Chapter 10, pp. 101, 106)
* **The 5% Extended Rule:**
  * *Exact Rule:* **Never buy a stock that is extended more than 5% past its ideal pivot/breakout price.**
  * *Reasoning:* Buying a stock 10% or 15% above the breakout point exposes the investor to normal 5% to 8% pullbacks or throwbacks that will trigger stop-loss liquidations even if the stock eventually goes on to make a major move.

#### Jack Schwager (*Getting Started in Technical Analysis*, Rule 8, p. 290 & Chapter 12, p. 164)
* **Rule 8 (Missed First Move):**
  * *Exact Wording:* *"Don't let the fact that you missed the first major portion of a new trend keep you from trading with that trend (as long as you can define a reasonable stop-loss point)."* (p. 290).
  * *Re-entry via Consolidation:* Schwager demonstrates that even after a massive price advance, if the market pauses and forms a new narrow consolidation (like a 3-day flag or tight trading range), you can safely enter with a tight stop placed just below the new consolidation boundary (Chapter 12, p. 164).

#### Thomas Bulkowski (*Encyclopedia of Chart Patterns*, Chapter 3, p. 48)
* **The 5% Overhead Resistance Rule:** If nearby overhead resistance or prior congestion is **less than 5% away** from the breakout price, do not enter the trade. The nearby resistance will retard momentum, increase throwback probability, and degrade the risk/reward ratio.

---

### 9. Handling Missed Trades & Resting Orders

#### Jack Schwager (*Getting Started in Technical Analysis*, Rules 5 & 19, p. 290–291 & Chapter 12, p. 162)
* **Rule 5 (Daily Review of Missed Entries):**
  * *Rule:* Record trade ideas from daily analysis and review them each day until entered or invalid. *"One common occurrence is that a trade idea is recalled once the market has moved beyond the intended entry, and it is then difficult to do the same trade at a worse price."* (p. 290).
* **Rule 19 (Order Cancellation on Consolidation):**
  * *Rule:* *"If selling into resistance or buying into support and the market consolidates instead of reversing, get out [and cancel resting orders]."* (p. 291). If price stalls at an S/R level rather than bouncing, it signals a breakout is forming against your order.
* **Salvaging Missed Trades (Chapter 12, p. 162):** If a limit order was placed in a consolidation (e.g., inside a triangle) but price breaks out before filling the limit order, Schwager advises: if the chart forms a secondary continuation pattern (like a flag), execute a **market order** on the secondary pattern to salvage the remaining profit potential rather than writing it off as a missed trade.

#### Martin Pring (*Technical Analysis Explained*, Chapter 9, p. 218)
* **Pring’s Discipline Rule for Missed Breakaway Gaps:**
  * *Exact Wording:* *"If you miss out because the price does not experience a retracement, all you have lost is an opportunity... at least you will not have lost any capital. With markets, there is always another opportunity. If you have the patience and the discipline to wait for that opportunity, you will be much better off in the long run."* (Chapter 9, p. 218). Pring strictly forbids emotional "chasing" of breakaway gaps.

---

### 10. Summary of Author Disagreements and Areas of No Guidance

| Topic | Author Agreements / Disagreements | Guidance Status |
| :--- | :--- | :--- |
| **Pullback Performance** | **Murphy & Schwager** note pullbacks offer better entry prices but risk missing moves. **Bulkowski's statistical data** proves that patterns with pullbacks underperform patterns without pullbacks across almost all pattern types. | Disagreement on desirability: Murphy/Schwager view pullbacks as a good compromise; Bulkowski's stats show they signal weaker momentum. |
| **Limit vs. Market Orders** | **Schwager (Rule 11)** strongly favors **Market Orders** for major entries to avoid missing trades. **Murphy** emphasizes using **Limit Orders** but placing them slightly offset from round numbers. | Different preferences based on trade horizon. |
| **Max Extended Limit** | **O'Neil** defines a strict **5% ceiling** past the pivot price. **Schwager** permits buying late in a trend provided a new consolidation forms to define a close stop. | Disagreement on strictness. |
| **Sub-minute Intraday Orders** | None of the books cover sub-minute HFT order routing, dark pools, or algorithmic market-if-touched execution mechanics. | **NO GUIDANCE in text** (all rules apply to daily/intraday bar charts, limit/market/stop orders). |

---

💡 **Next Step Suggestion:**
Would you like to synthesize these entry rules and filters into a single, step-by-step flowchart or decision tree for your intraday execution model?