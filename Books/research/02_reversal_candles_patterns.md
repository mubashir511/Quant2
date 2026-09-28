# reversal_candles_patterns

_NotebookLM answer, 2026-09-25 - paraphrase with page citations, UNVERIFIED until checked with tools.nlm.verify_quotes._

## Question

You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system for forex, metals, crypto, indices and stocks. Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree. Prefer concrete, mechanically testable rules (thresholds, counts, conditions) over general advice, and list any numbers the books actually give (percentages, bar counts, success rates).  QUESTION: What candlestick and chart patterns do the books say mark a price REVERSAL at a support or resistance region, and how reliable is each? Give for each pattern (hammer, hanging man, engulfing, piercing, dark cloud, morning/evening star, doji variants, harami, shooting star, tweezers, etc. and the reversal chart patterns such as double top/bottom, head and shoulders, wedges) its exact definition, the CONTEXT requirement (prior trend, location at S/R), the CONFIRMATION requirement (what the next bar must do), the stop location the author recommends, and any measured success/failure statistics (Bulkowski's failure rates, average moves, throwback/pullback rates). Which single patterns or pattern-plus-context combinations do the authors rate highest?

## Answer

Based on the books in your notebook, here is the complete breakdown of candlestick and chart reversal patterns, including definitions, context/location requirements, confirmation rules, stop-loss placements, and measured statistical reliability.

---

### Part I: Candlestick Reversal Patterns

#### 1. Hammer & Hanging Man (Takuri & Kubitsuri)
* **Exact Definition:**
  * Small real body (white or black) located at the upper end of the session's range.
  * Long lower shadow at least **2 to 3 times the height of the real body**.
  * Little or no upper shadow (*Nison, Ch. 4, pp. 33–34; Murphy, Ch. 12; Pring, Ch. 16, p. 288*).
* **Context Requirement:**
  * **Hammer:** Must come after a price decline/downtrend (*Nison, Ch. 4; Rockefeller, Ch. 8, p. 125*).
  * **Hanging Man:** Must come after a price rally/uptrend, preferably at an extended high or resistance (*Nison, Ch. 4*).
* **Confirmation Requirement:**
  * **Hammer:** Does *not* strictly require confirmation, though a white real body or a higher open/close on the next bar increases credibility (*Nison, Ch. 4, p. 34*).
  * **Hanging Man:** **MANDATORY confirmation required.** The next session must open or close *below* the hanging man’s real body (e.g., a long black candlestick) (*Nison, Ch. 4, p. 34; Ch. 12, p. 232; Pring, Ch. 16, p. 288*).
* **Stop Location:**
  * **Hammer:** Stop placed below the lowest point of the lower shadow (*Nison, Ch. 4, 11*).
  * **Hanging Man:** Stop placed above the high of the hanging man (*Nison, Ch. 4*).
* **Reliability / Statistics:**
  * Rockefeller (*Technical Analysis For Dummies*, Ch. 8, pp. 129–130, citing Bulkowski’s *Encyclopedia of Candlestick Charts*) notes the hammer works nearly all the time in FX markets, and ranks among Bulkowski's top "investment grade" candidates.

---

#### 2. Engulfing Patterns (Bullish & Bearish)
* **Exact Definition:**
  * **Bullish Engulfing:** A small black real body in a downtrend is completely covered/engulfed by a subsequent large white real body (the white open is below the prior black close, and the white close is above the prior black open) (*Nison, Ch. 3, 12; Murphy, Ch. 12; Pring, Ch. 16, p. 290*).
  * **Bearish Engulfing:** A small white real body in an uptrend is completely engulfed by a large black real body (*Nison, Ch. 3; Murphy, Ch. 12*).
* **Context Requirement:** Must occur after an established trend at a major support or resistance level (*Nison, Ch. 3, 11*).
* **Confirmation Requirement:** The two-candle structure itself forms the signal, but heavy volume on the second candle or engulfing multiple prior real bodies heightens its validity (*Nison, Ch. 3*).
* **Stop Location:**
  * **Bullish:** Stop placed below the low of the two-candle pattern (*Nison, Ch. 3, 11*).
  * **Bearish:** Stop placed above the high of the two-candle pattern (*Nison, Ch. 3*).
* **Reliability / Statistics:** Rockefeller (*Ch. 8, p. 129*) lists Bearish and Bullish Engulfing candles among the **6% of "investment grade" patterns** that deliver expected outcomes in >66% of trades.

---

#### 3. Piercing Pattern & Dark Cloud Cover
* **Exact Definition:**
  * **Piercing Pattern:** In a downtrend, a long black candle is followed by a gap-lower open on the next bar, which then rallies to close **more than 50% into the prior black candle's real body** (*Nison, Ch. 3, 12; Murphy, Ch. 12; Pring, Ch. 16, p. 290*).
  * **Dark Cloud Cover:** In an uptrend, a long white candle is followed by a gap-higher open, which then falls to close **more than 50% into the prior white candle's real body** (*Nison, Ch. 3, 12; Murphy, Ch. 12*).
* **Context Requirement:** Established trend at key support or resistance (*Nison, Ch. 3*).
* **Confirmation Requirement:** If the second candle fails to close at least 50% into the prior real body, it is classified as an In-Neck, On-Neck, or Thrusting line (which are bearish continuation patterns) (*Nison, Ch. 3, 7, p. 107*).
* **Stop Location:** Stop placed beyond the extreme low (Piercing) or extreme high (Dark Cloud Cover) (*Nison, Ch. 3*).

---

#### 4. Morning Star & Evening Star (and Doji Star Variants)
* **Exact Definition:**
  * **Morning Star (3 Bars):** 1) Extended black real body. 2) Small real body (star) gapping below the first body. 3) White real body intruding deeply into the first session's black body (*Nison, Ch. 5, pp. 62–65; Murphy, Ch. 12; Pring, Ch. 16, p. 291*).
  * **Evening Star (3 Bars):** 1) Extended white real body. 2) Small real body (star) gapping above the first body. 3) Black real body closing deeply into the first white body (*Nison, Ch. 5, pp. 66–69*).
  * **Doji Star Variants:** The center star candle is a Doji instead of a small spinning top (*Nison, Ch. 5, pp. 70–74*).
* **Context Requirement:** Prior trend at S/R zones (*Nison, Ch. 5*).
* **Confirmation Requirement:** The third candle must close deeply (>50%) into the first candle's body. If price moves beyond the star, the pattern is canceled (*Nison, Ch. 5; Pring, Ch. 16, p. 291*).
* **Stop Location:** Stop placed below the lowest low (Morning Star) or above the highest high (Evening Star) (*Nison, Ch. 5*).
* **Reliability / Statistics:** The **Bearish Doji Star** is rated by Bulkowski as an "investment grade" pattern with >66% reliability (*Rockefeller, Ch. 8, p. 129*).

---

#### 5. Doji Variants (Long-Legged, Gravestone, Dragonfly)
* **Exact Definition:** Open and Close are identical or nearly identical (*Nison, Ch. 8, p. 155*).
  * **Long-Legged Doji / Rickshaw Man:** Very long upper and lower shadows; open/close near the center (*Nison, Ch. 8, p. 161*).
  * **Gravestone Doji:** Open, Low, and Close are at the low of the session (*Nison, Ch. 8, p. 162*).
  * **Dragonfly Doji:** Open, High, and Close are at the high of the session (*Nison, Ch. 8, p. 165*).
* **Context Requirement:** Must appear after a distinct rally ("Northern Doji") or decline ("Southern Doji"). **Dojis inside a sideways trading range / "box" have NO forecasting implication** (*Nison, Ch. 8, pp. 165–166*).
* **Confirmation Requirement:** Dojis require confirmation from the next bar's close (*Nison, Ch. 8*).
* **Stop Location:** Above the upper shadow (Northern) or below the lower shadow (Southern) (*Nison, Ch. 8*).

---

#### 6. Harami & Harami Cross ("Petrifying Pattern")
* **Exact Definition:** Two-bar pattern where a small real body (Bar 2) is completely contained within the unusually large real body of Bar 1 (*Nison, Ch. 6, pp. 82–87*).
  * **Harami Cross:** Bar 2 is a Doji contained within Bar 1's real body (*Nison, Ch. 6, p. 83*).
* **Context Requirement:** Extended trend. Indicates the market is "losing its breath" (*Nison, Ch. 6*).
* **Confirmation Requirement:** Requires confirmation from the next bar's close beyond the small body (*Nison, Ch. 6; Rockefeller, Ch. 8*).
* **Stop Location:** Placed beyond the mother candle's high or low (*Nison, Ch. 6*).

---

#### 7. Shooting Star & Inverted Hammer
* **Exact Definition:** Small real body at the lower end of the range, long upper shadow (at least 2x real body height), little/no lower shadow (*Nison, Ch. 5, pp. 74–78*).
  * **Shooting Star:** Appears after a rally (bearish reversal).
  * **Inverted Hammer:** Appears after a decline (bullish reversal) (*Nison, Ch. 5, p. 78*).
* **Confirmation Requirement:**
  * **Shooting Star:** Best when confirmed by a black candle or gap down next session (*Nison, Ch. 5*).
  * **Inverted Hammer:** **REQUIRES bullish confirmation** on the next bar (a white candle closing higher) because the long upper shadow shows sellers pushed price back down (*Nison, Ch. 5, p. 78; Murphy, Ch. 12*).
* **Stop Location:** Stop placed above the tip of the upper shadow (Shooting Star) or below the low (Inverted Hammer) (*Nison, Ch. 5*).

---

#### 8. Tweezers (Tweezer Tops / Bottoms)
* **Exact Definition:** Two or more adjacent sessions with matching highs (Tweezer Top) or matching lows (Tweezer Bottom) (*Nison, Ch. 6, pp. 87–92*).
* **Context Requirement:** Occurs at major S/R levels (*Nison, Ch. 6*).
* **Confirmation Requirement:** Most potent when combined with another candlestick pattern (e.g., a Tweezer Top made of a tall white candle and a Hanging Man, or a Harami Cross) (*Nison, Ch. 6, p. 87*).
* **Stop Location:** Stop placed just beyond the matching highs or lows (*Nison, Ch. 6*).

---

### Part II: Reversal Chart Patterns & Bulkowski's Statistics

#### 1. Double Bottoms & Double Tops
* **Exact Definition:**
  * **Double Bottom:** Two distinct troughs at approximately the same price level separated by a reaction peak (the confirmation line) (*Bulkowski, Ch. 14–16; Murphy, Ch. 5; Rockefeller, Ch. 9*).
  * **Double Top:** Twin peaks at approximately the same price separated by a reaction valley (*Bulkowski, Ch. 17–20*).
* **Confirmation Requirement:**
  * **MUST close beyond the confirmation line** (peak for double bottom, valley for double top) (*Bulkowski, Ch. 14, 17; Rockefeller, Ch. 9, pp. 132–133*).
  * **Crucial Stat (Rockefeller, Ch. 9, pp. 131–133):** Unconfirmed twin bottom patterns fail ~2/3 of the time (65–67% failure rate). Once **CONFIRMED** by breaking the confirmation line, the Double Bottom achieves a **97% success rate** (only 3–5% break-even failure rate), and Double Tops achieve an **83% success rate**!
* **Stop Location:** Stop placed \$0.15 below the lowest bottom or above the highest top (*Bulkowski, Ch. 14, 17*).
* **Bulkowski's Statistics:**
  * **Double Bottoms (Eve & Eve / Adam & Eve):** Average rise = **35% to 37%** in bull markets. Break-even failure rate = **4% to 5%**. Throwback rate = **57% to 59%**. Target met = **66%**.
  * **Double Tops (Eve & Eve / Adam & Eve):** Average decline = **18% to 22%** (bull) / **24% to 25%** (bear). Break-even failure rate = **11% to 14%** (bull) / **2% to 7%** (bear). Pullback rate = **58% to 64%**. Target met = **69% to 73%**.

---

#### 2. Head and Shoulders Tops & Bottoms
* **Exact Definition:** Three peaks/troughs, with the middle extreme (head) higher/lower than the left and right shoulders (*Bulkowski, Ch. 24, 26; Murphy, Ch. 5; Pring, Ch. 5*).
* **Confirmation Requirement:** Daily close beyond the **Neckline** (*Bulkowski, Ch. 24, 26; Murphy, Ch. 5*).
* **Stop Location:** Placed beyond the right shoulder peak/low or the neckline (*Bulkowski, Ch. 24, 26*).
* **Bulkowski's Statistics:**
  * **Head & Shoulders Top:** Performance rank = **1 out of 21** in bull markets! Break-even failure rate = **4%** (bull) / **1%** (bear). Average decline = **22%** (bull) / **29%** (bear). Pullback rate = **50%** (bull) / **64%** (bear). Target met = **55% to 63%**.
  * **Head & Shoulders Bottom:** Break-even failure rate = **3%** (bull) / **4%** (bear). Average rise = **38%** (bull) / **31%** (bear). Throwback rate = **45%** (bull) / **51%** (bear). Target met = **74%**!

---

#### 3. Wedges (Falling & Rising Wedges)
* **Exact Definition:** Oscillating price between two converging trendlines that slope in the *same* direction (*Bulkowski, Ch. 52, 53; Murphy, Ch. 6*). Requires at least 5 touches (*Bulkowski, Ch. 52*).
  * **Falling Wedge:** Slopes down, breaks out UPWARD.
  * **Rising Wedge:** Slopes up, breaks out DOWNWARD.
* **Confirmation Requirement:** Close beyond the sloped trendline opposite the wedge direction (*Bulkowski, Ch. 52*).
* **Stop Location:** Placed beyond the extreme point of the wedge (*Bulkowski, Ch. 52*).
* **Bulkowski's Statistics:**
  * **Falling Wedge (Upward Breakout):** Average rise = **34%** (bull). Break-even failure rate = **10%**. Throwback rate = **63%**. Target met = **86%**!
  * **Rising Wedge (Downward Breakout):** Average decline = **17% to 28%**. Failure rate = **8% to 14%**.

---

#### 4. Bump-and-Run Reversal (BARR) Bottoms
* **Exact Definition:** Lead-in trendline, steep "bump" decline, and rally breaking the trendline (*Bulkowski, Ch. 7, pp. 114–122*).
* **Bulkowski's Statistics:** Performance rank = **8 out of 23** (bull) / **3 out of 19** (bear). Failure rate = **1% to 2%**. Average rise = **38%** (bull). Target met = **68%**.

---

### Part III: Which Patterns / Combinations Are Rated Highest?

#### 1. Single Highest-Rated Chart Patterns (Bulkowski Data)
1. **High and Tight Flags:** Average rise = **69%**, failure rate = **0%**, target met = **90%** (*Bulkowski, Ch. 22*).
2. **Head & Shoulders Tops:** Ranked **#1 out of 21** bearish patterns, failure rate = **1% to 4%** (*Bulkowski, Ch. 26*).
3. **Confirmed Double Bottoms:** Once confirmed by breaking the center line, achieves a **97% win rate** with an average gain of **40%** (*Rockefeller, Ch. 9, p. 132*).
4. **Falling Wedges:** Target met **86%** of the time (*Bulkowski, Ch. 52*).

#### 2. Top-Rated Candlestick Patterns
Out of 103 candlestick patterns tested across 10 years of data, Bulkowski found that only **13 patterns (6%) achieve "investment grade" status** (working as advertised in 66%+ of trades) (*Rockefeller, Ch. 8, p. 129*):
* **Bearish Doji Star**
* **Bearish Engulfing Candle**
* **Bullish Engulfing Candle**
* **Hammer / Shooting Star**
* **Rising and Falling Windows (Gaps)**

#### 3. Highest-Rated Pattern-Plus-Context Combinations (Confluence)
All authors explicitly state that **pattern + technical context** yields the highest probability set-ups:
* **Candlestick + S/R Zone + Oversold Oscillator:** Rockefeller (*Ch. 8, p. 127*) calls this **"a high-probability trade set-up, which means to get out the big guns because you want to buy!"** (e.g., Bullish Engulfing or Hammer at Support while RSI/Stochastics is oversold).
* **Eastern Candle + Western S/R (Nison's "Convergence"):** Steve Nison (*Part 2, Ch. 11–13*) identifies an **umbrella line (Hammer) or Engulfing pattern forming at a 50%/61.8% Fibonacci level or broken prior resistance (Change of Polarity)** as the single strongest reversal setup in technical analysis.

---

### Summary Table for System Design

| Pattern | Type | Context Requirement | Confirmation Requirement | Stop Loss Location | Reliability / Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hammer** | Candle (1-Bar) | Downtrend / Support | Optional (White body helps) | Below lower shadow low | High in FX; ~60-66% in equities |
| **Hanging Man** | Candle (1-Bar) | Uptrend / Resistance | **Mandatory** (Next bar close below body) | Above upper high | Requires confirmation bar |
| **Bullish/Bearish Engulfing** | Candle (2-Bar) | Trend / S/R zone | 2-candle structure; volume helps | Below low / Above high | **Top 13 "Investment Grade"** (>66%) |
| **Piercing / Dark Cloud** | Candle (2-Bar) | Trend / S/R zone | Close >50% into prior real body | Below low / Above high | High reliability at S/R |
| **Morning / Evening Star** | Candle (3-Bar) | Trend / S/R zone | 3rd candle closes >50% into 1st | Below star low / Above star high | **Bearish Doji Star is "Investment Grade"** |
| **Doji (Gravestone / Dragonfly)** | Candle (1-Bar) | Distinct trend (Northern/Southern) | **Mandatory** (Next bar direction) | Above/below shadow extreme | High at extremes; Useless in ranges |
| **Double Bottom (Confirmed)** | Chart Reversal | Downtrend | **Mandatory** (Close above center peak) | \$0.15 below lowest bottom | **97% Success Rate** post-breakout |
| **Head & Shoulders Top** | Chart Reversal | Uptrend | **Mandatory** (Close below Neckline) | Above right shoulder / Neckline | **#1 Rated Bear Pattern** (1-4% failure) |
| **Falling Wedge** | Chart Reversal | Downtrend / Bullish | Close above upper trendline | Below wedge low | **86% Target Met** rate; 10% failure |

---

💡 **Next Step Suggestion:** 
Would you like to explore how to combine these candlestick reversal triggers with specific intraday moving average rules (like the 10-week/50-day MAs or EMA ribbon crossovers) to refine your entry filters?