//+------------------------------------------------------------------+
//| Quant2ChartOverlay.mq5                                            |
//|                                                                    |
//| Reads the plain-text overlay file Quant2's Python side writes     |
//| (ai/chart_overlay.py, once per Clerk poll) and draws it directly  |
//| on this chart: support/resistance price ZONES (not hairlines --   |
//| Murphy's own "S/R is a zone, not an exact price" principle, using |
//| this account's own real swing-clustering tolerance), a native     |
//| MT5 Fibonacci retracement tool anchored at the real swing points   |
//| with each ratio band individually labeled (including calling out  |
//| the 38.2-61.8% "golden zone" pullback region this account's own   |
//| setup classifier already treats as the real entry zone), a shaded |
//| "realistic same-session move" band (H1-ATR based), trendlines,    |
//| entry/stop/target lines, full-height regime-transition markers    |
//| (trending/sideways/accumulation/distribution), and short text     |
//| notes -- direct user request 2026-09-04: "I usually not read long |
//| prompts and reasoning, I am comfortable to read lines, levels,    |
//| regions and some notes on the mt5 terminal screen."               |
//|                                                                    |
//| v3 (2026-09-04, direct user feedback on the v2 drawing):          |
//|  1. Colors now ADAPT to the chart's own background (light vs dark |
//|     theme), read live via CHART_COLOR_BACKGROUND -- v2 hardcoded  |
//|     a dark-theme palette (white notes, pale silver/goldenrod)     |
//|     that was unreadable on a white chart background.              |
//|  2. Every label now carries real, short interpretive context (not |
//|     just "H4 resistance, 3x touched" but what it means and what   |
//|     to watch for), generated in ai/chart_overlay.py alongside the |
//|     real numbers.                                                  |
//|  3. A new LOG record type carries this account's own real,        |
//|     already-computed thesis/side/entry-stop-target as one line,   |
//|     Print()'d into the Experts log as a "strategy briefing" -- a  |
//|     real update on what's actually being watched and why, not     |
//|     just a health-check line count.                               |
//|                                                                    |
//| v4: real low-opacity fills via the Canvas (ARGB bitmap) class --  |
//| MT5's plain OBJ_RECTANGLE has no alpha support at all, so the     |
//| earlier hollow-border-only zones are now ALSO tinted, without     |
//| ever hiding the candles. Fibonacci switched to a heat-map level   |
//| palette (hot at the 38.2-61.8% golden zone, cool at the anchors), |
//| and its own native diagonal 0%-100% connector line (a geometric   |
//| artifact of the tool, not a real trendline) is hidden by coloring |
//| it to match the chart background.                                 |
//|                                                                    |
//| v6 (2026-09-04, direct user feedback on the v4/v5 drawing):       |
//|  1. Every Fibonacci band (the fill between each pair of adjacent  |
//|     ratios) now carries its own visible label naming its range -- |
//|     several unlabeled colored regions at once ("green, red, blue, |
//|     orange... don't know what represents what") was the direct    |
//|     complaint. The separate GOLDENZONE rectangle (v2) is gone --  |
//|     it duplicated two of these bands' own price range with a      |
//|     second, differently-labeled overlapping tint.                 |
//|  2. RefreshOverlay now only clears and redraws when the file's own |
//|     content for THIS symbol has genuinely changed since the last  |
//|     poll (a content fingerprint comparison) -- direct complaint:  |
//|     "the region flashes/jerk on every refresh." The old code       |
//|     unconditionally cleared and redrew everything every single    |
//|     RefreshSeconds tick even when nothing about this symbol had    |
//|     changed. A chart scroll/zoom now only repositions the         |
//|     screen-pixel-anchored canvas fills (see RepositionCanvasesOnly |
//|     and OnChartEvent) rather than triggering a full redraw of      |
//|     everything else, which needs no such help from MT5 anyway.    |
//|                                                                    |
//| v7 (2026-09-04, direct user feedback again): the v6 per-band       |
//| labels fixed the ambiguity but made the chart itself "messy and    |
//| unprofessional" -- removed EVERY permanent on-chart region/line     |
//| label (zones, Fibonacci bands, trendlines) in favor of ONE compact |
//| color-key legend (DrawLegend, top-right corner): each row's own    |
//| text color IS the swatch, one or two words each, no sentences. The |
//| removed detail still surfaces on hover via OBJPROP_TOOLTIP, so     |
//| nothing is actually lost, just no longer permanently on screen.    |
//| Entry/Stop/Target keep their own short on-chart labels (already    |
//| one or two words, and essential to see without hovering).          |
//|                                                                    |
//| v8 (2026-09-04, direct user request, same day): full-height        |
//| vertical lines (OBJ_VLINE) marking real regime TRANSITIONS --      |
//| trending/sideways/accumulation/distribution -- driven by a genuine |
//| new backend computation (analysis.technical.compute_regime_segments|
//| ), not just a drawing tweak: TechnicalStats.market_regime was only |
//| ever a single latest-window snapshot with no memory of when the    |
//| current regime began or what preceded it.                          |
//|                                                                    |
//| v9 (2026-09-04, direct user feedback on the v8 legend/vlines):     |
//|  1. The legend is now ONE bitmap panel (DrawLegendPanel) with a    |
//|     real background fill behind its text instead of bare OBJ_LABEL |
//|     rows floating over the chart -- direct complaint: legend text  |
//|     was mixing into whatever chart content sat behind it. Drawn    |
//|     LAST in every redraw pass so it's guaranteed on top of every   |
//|     zone/line, not just usually on top.                            |
//|  2. Each regime segment's own SPAN (not just its boundary) is now  |
//|     shaded (DrawRegimeRegion) -- direct request: "add distinct     |
//|     hatches in the vertical regions as well." The boundary line    |
//|     (v8) stays; the shading is additive.                           |
//|                                                                    |
//| v10 (2026-09-04, direct user feedback again): the legend text was  |
//| still too small to read comfortably -- bumped its own font size    |
//| (LEGEND_FONT_SIZE). The v9 regime-region shading switched from a   |
//| directional hatch texture to a flat, low-opacity fill instead --   |
//| direct request, same visual language as the existing S/R/Fib fills |
//| rather than a distinct texture.                                    |
//|                                                                    |
//| v11 (2026-09-05, direct user feedback): 1. Every tooltip-bearing   |
//| line/border (trendlines, hard lines, zone borders, regime vlines,  |
//| the Fibonacci tool) is now explicit BACK=false -- direct feedback  |
//| that hovering these still showed no description; MT5 appears to    |
//| exclude BACK=true objects from its own tooltip hit-test scan       |
//| entirely. 2. H1 vs H4 trendlines now use different dash styles,    |
//| not just different colors. 3. "Trending" regime split back into    |
//| separate Uptrend/Downtrend colors. 4. Fibonacci tick lines/text    |
//| switched from the heat-map palette to a plain black/white          |
//| (g_noteColor) -- the band FILLS keep the heat-map tint.            |
//|                                                                    |
//| Attach to any symbol's chart (designed around M5, but any period  |
//| works). Only draws the lines whose own symbol column matches THIS |
//| chart's _Symbol -- the file covers every symbol the account is    |
//| watching, filtered here per chart.                                 |
//|                                                                    |
//| Reads from <Common Data Folder>\Files\quant2_overlay.txt          |
//| (FILE_COMMON), not the per-terminal Files folder, so this finds   |
//| the file regardless of which account/terminal profile wrote it -- |
//| this account runs more than one MT5 login against the same        |
//| terminal.exe (see data/mt5_source.py's own FTMO_MT5_* vs plain     |
//| MT5_* config split).                                              |
//|                                                                    |
//| Pure display: never places, modifies, or closes a single order.   |
//| A missing/unreadable/malformed file just means nothing is drawn   |
//| this refresh, not an error -- Python's own write_chart_overlay    |
//| already degrades the same way on its side.                        |
//+------------------------------------------------------------------+
#property copyright "Quant2"
#property strict

#include <Canvas\Canvas.mqh>

input int RefreshSeconds = 5;

const string InputFileName = "quant2_overlay.txt";
const string OBJ_PREFIX = "Quant2_";

// Theme-adaptive palette -- populated by DetectTheme() every refresh
// (cheap; lets a live theme switch be picked up without re-attaching).
color g_supportColor, g_resistanceColor, g_entryColor, g_stopColor, g_targetColor;
color g_fiboColor, g_atrZoneColor, g_noteColor, g_trendColor;
// Regime-transition vertical markers (see DrawRegimeLine/RegimeColor) --
// 5 colors: uptrend/downtrend split apart with their own colors (direct
// user request 2026-09-05: "use different color for downtrend and
// uptrend markets" -- an earlier round had merged both into one
// "trending" swatch), plus sideways/accumulation/distribution. Still
// not the full 7 raw labels compute_regime_segments can return --
// choppy_up/down fold into their own direction's color, not a 6th/7th.
color g_uptrendRegimeColor, g_downtrendRegimeColor, g_sidewaysRegimeColor, g_accumulationColor, g_distributionColor;
// The chart's own live background color -- used to make the native
// OBJ_FIBO tool's own diagonal 0%-to-100% connector line INVISIBLE (see
// DrawFibo below) by coloring it identically to whatever's behind it,
// since that connector is a geometric artifact of the tool, not a real
// trendline, and direct user feedback called it out as "weird... I
// don't see they follow any pattern."
color g_chartBackgroundColor;

// Real semi-transparent fills -- direct user request 2026-09-04: "use
// very low transparency so that the chart candles are clearly
// visible." MT5's plain chart objects (OBJ_RECTANGLE etc.) have NO
// alpha-blending support at all; a "filled" rectangle is always fully
// opaque. The only genuine way to get a real translucent region on an
// MT5 chart is the Canvas class (an ARGB bitmap object composited over
// the chart), which is what every published "transparent S/R zone"
// indicator actually uses under the hood. A fixed pool of CCanvas
// instances (rather than one created/destroyed per rectangle every
// refresh) avoids repeated alloc/free churn every 5 seconds.
#define MAX_CANVASES 64
// Regime-region fill (direct user request 2026-09-04: a flat, low-
// opacity color fill across the whole segment's span, same visual
// language as the S/R/Fib fills below -- an earlier round tried a
// directional hatch texture here instead; direct feedback replaced it
// with this simpler flat wash). Deliberately LOWER than ZONE_FILL_ALPHA/
// FIBO_BAND_FILL_ALPHA (direct user request: vertical regions must read
// fainter than the horizontal zone/band fills) -- these vertical
// regions can also span a much larger chunk of the chart's own width (a
// regime segment can last many bars) than a small horizontal S/R zone
// ever does, so a large area needs a noticeably lower alpha just to
// avoid reading as a heavier tint overall despite the lower number.
#define REGIME_FILL_ALPHA 15
CCanvas g_canvasPool[MAX_CANVASES];
bool    g_canvasActive[MAX_CANVASES];
int     g_canvasCount = 0;

// Remembers every canvas fill's own (price, time, color, alpha) from
// the last real content redraw, so a chart scroll/zoom can reposition
// just the canvases (see RepositionCanvasesOnly) WITHOUT re-parsing the
// file or touching any of the other, natively price/time-anchored
// objects (HLINE/RECTANGLE-border/TREND/FIBO/TEXT need no such help --
// MT5 already tracks their positions on its own).
struct CanvasSpec
  {
   string   name;
   double   priceLow, priceHigh;
   datetime timeLow, timeHigh;
   color    clr;
   uchar    alpha;
  };
CanvasSpec g_canvasSpecs[MAX_CANVASES];
int        g_canvasSpecCount = 0;

// The 7 real Fibonacci ratios this account's own analysis actually uses
// (must match analysis/chart_structure.py's FIB_RATIOS = (0.0, 0.236,
// 0.382, 0.5, 0.618, 0.786, 1.0) -- a rarely-changed constant, hardcoded
// here rather than round-tripped through the overlay file to keep the
// line format simple). MT5's native OBJ_FIBO ships with its OWN default
// level set/colors (including 161.8/261.8 extensions this account never
// computes) that ignore whatever OBJPROP_COLOR is set on the object as a
// whole -- each ratio line has its own separate OBJPROP_LEVELCOLOR, which
// is what was actually rendering as MT5's default bright yellow,
// unreadable on a white chart, regardless of the "main" color set below.
// g_fiboLevelColors is a heat map, not an arbitrary palette: COOL at the
// two anchors (0%/100%, less actionable) and HOT in the middle -- the
// 38.2-61.8% golden zone this account's own setup classifier treats as
// the real pullback-entry area -- so the eye is drawn to the zone that
// actually matters, not just prettier colors.
const double g_fiboRatios[7] = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0};
color g_fiboLevelColors[7];

// A DEDICATED canvas for the legend panel -- separate from g_canvasPool
// (which holds this refresh's zone/fib/regime fills, all destroyed and
// rebuilt together every redraw) because the legend is always exactly
// one instance, drawn last, and must survive being the final thing
// composited on screen regardless of how many other canvases this
// symbol's own data happens to need that poll.
CCanvas g_legendCanvas;
bool    g_legendCanvasActive = false;

//+------------------------------------------------------------------+
int OnInit()
  {
   PrintFormat("Quant2ChartOverlay: attached to %s, refreshing every %ds. Reading %s (FILE_COMMON).",
               _Symbol, MathMax(1, RefreshSeconds), InputFileName);
   EventSetTimer(MathMax(1, RefreshSeconds));
   RefreshOverlay();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ClearCanvases();
   ClearLegendCanvas();
   ClearOverlay();
   ChartRedraw(0);
  }

void OnTimer()
  {
   RefreshOverlay();
  }

// Repositions ONLY the canvas fills on scroll/zoom/resize -- they're
// positioned in absolute screen PIXELS recomputed from the chart's
// current time/price axes, so without this they'd visibly lag behind
// the candles until the next content-changed redraw. Deliberately does
// NOT call the full RefreshOverlay(): every other object type (HLINE/
// RECTANGLE-border/TREND/FIBO/TEXT) is natively time/price-anchored by
// MT5 and needs no help at all, and RefreshOverlay's own content check
// (see its own docstring) would otherwise skip the canvases too, since
// a mere scroll never changes the FILE's content.
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(id == CHARTEVENT_CHART_CHANGE)
      RepositionCanvasesOnly();
  }

//+------------------------------------------------------------------+
// Reads the chart's own background color and picks a palette that
// reads clearly against it -- direct user feedback: a dark-theme
// palette (white notes, pale silver/goldenrod lines) was unreadable on
// a white chart with red/green candles. `color` in MQL5 packs as
// 0x00BBGGRR (the same layout as a Windows COLORREF), so bits 0-7 are
// red, 8-15 green, 16-23 blue.
void DetectTheme()
  {
   long bg = ChartGetInteger(0, CHART_COLOR_BACKGROUND);
   g_chartBackgroundColor = (color)(bg & 0xFFFFFF);
   int r = (int)(bg & 0xFF);
   int g = (int)((bg >> 8) & 0xFF);
   int b = (int)((bg >> 16) & 0xFF);
   double luminance = 0.299 * r + 0.587 * g + 0.114 * b;
   bool isLight = luminance > 150.0;

   if(isLight)
     {
      // Deeper, more saturated tones -- readable on white, and
      // deliberately distinct from the chart's own light-red/light-
      // green candle body colors so lines don't blend into candles.
      g_supportColor    = clrDarkGreen;
      g_resistanceColor = clrFireBrick;
      g_entryColor      = clrNavy;
      g_stopColor       = clrDarkRed;
      g_targetColor     = clrDarkGreen;
      g_fiboColor       = clrDarkOrange;
      g_atrZoneColor    = clrRoyalBlue;
      g_noteColor       = clrBlack;
      g_trendColor      = clrIndigo;
      // Heat map: cool RoyalBlue/SteelBlue at the 0%/100% anchors,
      // hot DarkGoldenrod at the 50% heart of the golden zone, DarkOrange
      // on either side of it (38.2%/61.8%) -- all deliberately deep/
      // saturated tones, never the pale/light shades that vanish on
      // white. DarkGoldenrod (was Crimson) -- direct user feedback:
      // Crimson read as "the same color" as Resistance zone's own
      // FireBrick once both were on screen at once; a genuine gold/
      // brown tone reads as visually distinct from every red already in
      // this palette (FireBrick/resistance) while still fitting "golden
      // zone" thematically. Not plain Yellow here specifically -- too
      // pale to stay readable on a white chart background, the exact
      // failure mode an earlier round already fixed once for this same
      // tool's default level colors.
      g_fiboLevelColors[0] = clrRoyalBlue;   // 0.0%
      g_fiboLevelColors[1] = clrSteelBlue;   // 23.6%
      g_fiboLevelColors[2] = clrDarkOrange;  // 38.2% (golden zone edge)
      g_fiboLevelColors[3] = clrDarkGoldenrod; // 50.0% (golden zone heart)
      g_fiboLevelColors[4] = clrDarkOrange;  // 61.8% (golden zone edge)
      g_fiboLevelColors[5] = clrSteelBlue;   // 78.6%
      g_fiboLevelColors[6] = clrRoyalBlue;   // 100.0%
      // Deliberately distinct hues from every OTHER color already
      // claimed above (not just from each other) -- direct user
      // feedback: SeaGreen (accumulation) read as "the same green" as
      // DarkGreen (support/target) once both were on screen at once, and
      // an earlier round's SlateGray/Gray pairing had the same problem
      // within the regime colors themselves. Now: cyan-teal, olive
      // (yellow-green), neutral gray, magenta-purple, deep brown -- five
      // families untouched by any other row in this legend.
      g_uptrendRegimeColor   = clrDarkCyan;
      g_downtrendRegimeColor = clrOlive;
      g_sidewaysRegimeColor  = clrGray;
      g_accumulationColor    = clrDarkMagenta;
      g_distributionColor    = clrSaddleBrown;
     }
   else
     {
      g_supportColor    = clrLimeGreen;
      g_resistanceColor = clrOrangeRed;
      g_entryColor      = clrDodgerBlue;
      g_stopColor       = clrRed;
      g_targetColor     = clrLimeGreen;
      g_fiboColor       = clrGoldenrod;
      g_atrZoneColor    = clrDodgerBlue;
      g_noteColor       = clrWhite;
      g_trendColor      = clrLightSkyBlue;
      // Same heat-map shape, brighter tones so it still pops on black.
      // Yellow (was Red) at the golden-zone heart, same reasoning as the
      // light-theme block above -- Red read as "the same color" as
      // Resistance zone's own OrangeRed. Pure Yellow is fine here
      // (unlike on a white chart) since a black background is exactly
      // where a pale, saturated tone reads clearly instead of vanishing.
      g_fiboLevelColors[0] = clrDeepSkyBlue;    // 0.0%
      g_fiboLevelColors[1] = clrCornflowerBlue; // 23.6%
      g_fiboLevelColors[2] = clrOrange;         // 38.2% (golden zone edge)
      g_fiboLevelColors[3] = clrYellow;         // 50.0% (golden zone heart)
      g_fiboLevelColors[4] = clrOrange;         // 61.8% (golden zone edge)
      g_fiboLevelColors[5] = clrCornflowerBlue; // 78.6%
      g_fiboLevelColors[6] = clrDeepSkyBlue;    // 100.0%
      // Same reasoning as the light-theme block above: brighter tones
      // so they still pop on a black background, and the SAME 5
      // genuinely separate hue families untouched by any other row.
      g_uptrendRegimeColor   = clrCyan;
      g_downtrendRegimeColor = clrYellowGreen;
      g_sidewaysRegimeColor  = clrSilver;
      g_accumulationColor    = clrMagenta;
      g_distributionColor    = clrChocolate;
     }
  }

//+------------------------------------------------------------------+
void ClearOverlay()
  {
   int total = ObjectsTotal(0, -1, -1);
   for(int i = total - 1; i >= 0; i--)
     {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, OBJ_PREFIX) == 0)
         ObjectDelete(0, name);
     }
  }

// Destroys every active canvas through its OWN API (frees its pixel
// buffer properly) rather than relying on ClearOverlay's generic
// ObjectDelete-by-name sweep, which would remove the underlying chart
// object but leave each CCanvas instance's internal state stale.
void ClearCanvases()
  {
   for(int i = 0; i < g_canvasCount; i++)
     {
      if(g_canvasActive[i])
        {
         g_canvasPool[i].Destroy();
         g_canvasActive[i] = false;
        }
     }
   g_canvasCount = 0;
  }

// Same reasoning as ClearCanvases, for the legend's own dedicated
// canvas -- frees its pixel buffer through its own API rather than
// leaving the CCanvas instance's internal state stale after
// ClearOverlay's generic ObjectDelete sweep removes the underlying
// chart object.
void ClearLegendCanvas()
  {
   if(g_legendCanvasActive)
     {
      g_legendCanvas.Destroy();
      g_legendCanvasActive = false;
     }
  }

// Packs an MQL5 color (0x00BBGGRR) plus an alpha byte into the
// 0xAARRGGBB layout CCanvas expects for COLOR_FORMAT_ARGB_NORMALIZE.
uint ToARGB(const color clr, const uchar alpha)
  {
   uint rr = (uint)(clr & 0xFF);
   uint gg = (uint)((clr >> 8) & 0xFF);
   uint bb = (uint)((clr >> 16) & 0xFF);
   return ((uint)alpha << 24) | (rr << 16) | (gg << 8) | bb;
  }

// Does the actual pixel-level draw: a real, low-opacity filled
// rectangle over the given (time, price) region -- candles underneath
// stay clearly visible through it. Screen pixel position is recomputed
// from the chart's CURRENT time/price axes every call, so calling this
// again (from RepositionCanvasesOnly) after a scroll/zoom re-anchors it
// correctly with no other state needed.
void RenderCanvasRect(const string name, const double priceLow, const double priceHigh,
                       const datetime timeLow, const datetime timeHigh, const color clr, const uchar alpha)
  {
   if(g_canvasCount >= MAX_CANVASES)
      return;

   int x1, y1, x2, y2;
   if(!ChartTimePriceToXY(0, 0, timeLow, priceHigh, x1, y1))
      return;
   if(!ChartTimePriceToXY(0, 0, timeHigh, priceLow, x2, y2))
      return;

   int left   = MathMin(x1, x2);
   int top    = MathMin(y1, y2);
   int width  = (int)MathMin(MathMax(2, MathAbs(x2 - x1)), 4000);
   int height = (int)MathMin(MathMax(2, MathAbs(y2 - y1)), 4000);

   int slot = g_canvasCount;
   if(!g_canvasPool[slot].CreateBitmapLabel(name, left, top, width, height, COLOR_FORMAT_ARGB_NORMALIZE))
      return;
   g_canvasActive[slot] = true;
   g_canvasCount++;

   g_canvasPool[slot].Erase(0x00000000);
   g_canvasPool[slot].FillRectangle(0, 0, width - 1, height - 1, ToARGB(clr, alpha));
   g_canvasPool[slot].Update();
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
  }

// Public entry point used while parsing the overlay file: draws the
// fill immediately AND records its own spec so a later scroll/zoom can
// reposition it (see RepositionCanvasesOnly) without needing to
// re-parse the file at all.
void DrawTransparentRect(const string name, const double priceLow, const double priceHigh,
                          const datetime timeLow, const datetime timeHigh, const color clr, const uchar alpha)
  {
   RenderCanvasRect(name, priceLow, priceHigh, timeLow, timeHigh, clr, alpha);
   if(g_canvasSpecCount < MAX_CANVASES)
     {
      g_canvasSpecs[g_canvasSpecCount].name      = name;
      g_canvasSpecs[g_canvasSpecCount].priceLow  = priceLow;
      g_canvasSpecs[g_canvasSpecCount].priceHigh = priceHigh;
      g_canvasSpecs[g_canvasSpecCount].timeLow   = timeLow;
      g_canvasSpecs[g_canvasSpecCount].timeHigh  = timeHigh;
      g_canvasSpecs[g_canvasSpecCount].clr       = clr;
      g_canvasSpecs[g_canvasSpecCount].alpha     = alpha;
      g_canvasSpecCount++;
     }
  }

// Redraws every remembered canvas fill at its current pixel position,
// touching nothing else -- see OnChartEvent's own docstring for why
// this must stay separate from the full content-driven redraw.
void RepositionCanvasesOnly()
  {
   if(g_canvasSpecCount == 0)
      return;
   ClearCanvases();
   int specCount = g_canvasSpecCount; // ClearCanvases doesn't touch this array; snapshot before the loop re-populates it
   for(int i = 0; i < specCount; i++)
      RenderCanvasRect(g_canvasSpecs[i].name, g_canvasSpecs[i].priceLow, g_canvasSpecs[i].priceHigh,
                        g_canvasSpecs[i].timeLow, g_canvasSpecs[i].timeHigh, g_canvasSpecs[i].clr, g_canvasSpecs[i].alpha);
   ChartRedraw(0);
  }

color LevelColor(const string kind)
  {
   if(kind == "support")
      return g_supportColor;
   if(kind == "resistance")
      return g_resistanceColor;
   return g_trendColor;
  }

color RegimeColor(const string category)
  {
   if(category == "uptrend")
      return g_uptrendRegimeColor;
   if(category == "downtrend")
      return g_downtrendRegimeColor;
   if(category == "accumulation")
      return g_accumulationColor;
   if(category == "distribution")
      return g_distributionColor;
   return g_sidewaysRegimeColor; // "sideways", or an unrecognized future category
  }

// A hard trade level (entry/stop/target) -- a solid line, colored per
// the legend (DrawLegend), with no permanent on-chart text of its own
// -- direct user request. Detail is still on the line's own hover
// tooltip. Foreground (BACK=false) -- see DrawTrend's own comment for
// why every tooltip-bearing line/border in this file now sets this
// explicitly.
void DrawHardLine(const string objBase, const double price, const color clr, const string text)
  {
   string lineName = objBase + "_ln";
   ObjectCreate(0, lineName, OBJ_HLINE, 0, 0, price);
   ObjectSetInteger(0, lineName, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, lineName, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, lineName, OBJPROP_BACK, false);
   ObjectSetInteger(0, lineName, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, lineName, OBJPROP_TOOLTIP, text);
  }

// Real alpha values for the low-transparency fills -- deliberately low
// (well under half) so candles stay clearly visible through them, per
// direct user request. Fib inter-level bands use a slightly lower
// value than the single-region fills since up to 6 of them can sit
// side by side and would otherwise visually compound.
#define ZONE_FILL_ALPHA 45
#define FIBO_BAND_FILL_ALPHA 32

// Support/resistance drawn as a real ZONE (a thin band, not a hairline)
// -- Murphy's own "support/resistance is an area, not an exact price"
// principle, using the SAME clustering tolerance this account's own
// analysis/chart_structure.py used to decide these swing points belong
// to the same level in the first place, not an arbitrary new number.
// A crisp dotted border marks the exact edges; a real, low-opacity fill
// (see DrawTransparentRect) tints the interior without ever hiding the
// candles behind it. No permanent on-chart text -- direct user
// feedback: per-region labels made the chart "messy and unprofessional"
// (see DrawLegend for the one-time color key instead); the full detail
// still surfaces on hover via OBJPROP_TOOLTIP.
// Split into a Fill half (drawn in RefreshOverlay's PASS 1, with every
// other fill) and a Border half (PASS 2, with every other line/border)
// -- direct user request 2026-09-04: "show the vertical lines, trend
// lines and region horizontal boxes over the hatch fill color... line
// should be at the front." MT5 stacks same-layer objects by creation
// order, so drawing every fill FIRST and every border/line SECOND, each
// pass across the WHOLE symbol rather than one zone at a time, is what
// actually guarantees borders/lines end up visually on top of every
// fill, not just their own.
void DrawLevelZoneFill(const string objBase, const double price, const double tolerancePct, const color clr)
  {
   double halfWidth = price * tolerancePct / 100.0;
   double low = price - halfWidth;
   double high = price + halfWidth;
   datetime t1 = iTime(_Symbol, PERIOD_CURRENT, 60);
   datetime t2 = TimeCurrent() + PeriodSeconds(PERIOD_CURRENT) * 10;
   DrawTransparentRect(objBase + "_fill", low, high, t1, t2, clr, ZONE_FILL_ALPHA);
  }

// Support/resistance drawn as a real ZONE (a thin band, not a hairline)
// -- Murphy's own "support/resistance is an area, not an exact price"
// principle, using the SAME clustering tolerance this account's own
// analysis/chart_structure.py used to decide these swing points belong
// to the same level in the first place, not an arbitrary new number.
// A crisp dotted border marks the exact edges; the real, low-opacity
// fill (DrawLevelZoneFill, PASS 1) tints the interior without ever
// hiding the candles behind it. No permanent on-chart text -- direct
// user feedback: per-region labels made the chart "messy and
// unprofessional" (see DrawLegendPanel for the one-time color key
// instead); the full detail still surfaces on hover via OBJPROP_TOOLTIP.
void DrawLevelZoneBorder(const string objBase, const double price, const double tolerancePct, const color clr, const string label)
  {
   double halfWidth = price * tolerancePct / 100.0;
   double low = price - halfWidth;
   double high = price + halfWidth;
   datetime t1 = iTime(_Symbol, PERIOD_CURRENT, 60);
   datetime t2 = TimeCurrent() + PeriodSeconds(PERIOD_CURRENT) * 10;
   string rectName = objBase + "_zone";
   ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, t1, high, t2, low);
   ObjectSetInteger(0, rectName, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, rectName, OBJPROP_FILL, false); // border only -- the real fill is the canvas drawn in PASS 1
   ObjectSetInteger(0, rectName, OBJPROP_BACK, false); // foreground -- see DrawTrend's own comment
   ObjectSetInteger(0, rectName, OBJPROP_STYLE, STYLE_DOT);
   ObjectSetInteger(0, rectName, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, rectName, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, rectName, OBJPROP_TOOLTIP, label);
  }

void DrawRegionFill(const string objBase, const double low, const double high, const color clr)
  {
   datetime t1 = iTime(_Symbol, PERIOD_CURRENT, 40);
   datetime t2 = TimeCurrent() + PeriodSeconds(PERIOD_CURRENT) * 10;
   DrawTransparentRect(objBase + "_fill", low, high, t1, t2, clr, ZONE_FILL_ALPHA);
  }

// A generic region (currently the ATR "realistic move" band) -- solid
// border plus the real, low-opacity fill (DrawRegionFill, PASS 1), same
// reasoning as DrawLevelZoneBorder. No permanent on-chart text -- see
// its own comment above; detail is on the border's own hover tooltip.
void DrawRegionBorder(const string objBase, const double low, const double high, const color clr, const string label)
  {
   datetime t1 = iTime(_Symbol, PERIOD_CURRENT, 40);
   datetime t2 = TimeCurrent() + PeriodSeconds(PERIOD_CURRENT) * 10;
   string rectName = objBase + "_zone";
   ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, t1, high, t2, low);
   ObjectSetInteger(0, rectName, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, rectName, OBJPROP_FILL, false);
   ObjectSetInteger(0, rectName, OBJPROP_BACK, false); // foreground -- see DrawTrend's own comment
   ObjectSetInteger(0, rectName, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, rectName, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, rectName, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, rectName, OBJPROP_TOOLTIP, label);
  }

// H4's own trendlines use a longer secondsBack than H1's (see
// ai/chart_overlay.py's _TREND_LOOKBACK_BARS * _TIMEFRAME_SECONDS: H4 =
// 20*14400=288000s, H1 = 20*3600=72000s) -- a real, existing signal for
// which timeframe a given TREND record belongs to, with no new overlay
// field needed. 150000s sits cleanly between the two.
#define H4_TREND_SECONDS_THRESHOLD 150000

// Anchored with REAL elapsed seconds (TimeCurrent() - secondsBack),
// never a chart bar-offset -- the ANALYSIS timeframe's own bar spacing
// (H4 = 4h/bar, H1 = 1h/bar) is what the price math is based on, so
// this must anchor in real time, not this chart's own bar count, to
// stay correct regardless of what period the chart is set to. No
// permanent on-chart text -- see DrawLevelZone's own comment above;
// detail is on the line's own hover tooltip. Foreground (BACK=false) --
// direct user feedback: hovering a BACK=true line never showed its
// tooltip; MT5 appears to exclude background-layer objects from its own
// tooltip hit-test scan entirely (matches the same fix already applied
// to DrawRegimeRegion's own hit-test rectangle).
void DrawTrend(const string objBase, const double price1, const double price2, const int secondsBack, const string kind, const string label)
  {
   datetime t1 = TimeCurrent() - secondsBack;
   datetime t2 = TimeCurrent();
   string lineName = objBase + "_ln";
   ObjectCreate(0, lineName, OBJ_TREND, 0, t1, price1, t2, price2);
   ObjectSetInteger(0, lineName, OBJPROP_COLOR, LevelColor(kind));
   // Direct user request 2026-09-05: "use different dashed style for
   // different trend lines" -- H1 vs H4 now render with visually
   // distinct dash patterns (on top of the existing support/resistance
   // color difference), so all 4 possible trendlines on one chart read
   // as genuinely different lines, not just 2 colors repeated twice.
   bool isH4 = secondsBack >= H4_TREND_SECONDS_THRESHOLD;
   ObjectSetInteger(0, lineName, OBJPROP_STYLE, isH4 ? STYLE_DASHDOTDOT : STYLE_DASH);
   ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, lineName, OBJPROP_RAY_RIGHT, true); // project forward into current price action
   ObjectSetInteger(0, lineName, OBJPROP_BACK, false);
   ObjectSetInteger(0, lineName, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, lineName, OBJPROP_TOOLTIP, label);
  }

// Full-height vertical marker at a real regime TRANSITION -- direct
// user request 2026-09-04: "add top to bottom straight lines to
// indicate accumulation, distribution regions, trending and sideways
// regimes." OBJ_VLINE natively spans the whole chart height and is
// time-anchored (auto-repositions on scroll/zoom exactly like every
// other native object here, no canvas needed). Anchored with REAL
// elapsed seconds -- same reasoning as DrawTrend/DrawFibo above, since
// analysis.technical.compute_regime_segments works off H1 bars
// regardless of what period this chart itself is set to. No permanent
// on-chart text, matching this file's v7 convention -- detail is on the
// line's own hover tooltip; its color (via RegimeColor) is the only
// thing distinguishing categories, keyed in DrawLegend.
void DrawRegimeLine(const string objBase, const int secondsAgo, const string category, const string label)
  {
   datetime t = TimeCurrent() - secondsAgo;
   string lineName = objBase + "_vln";
   ObjectCreate(0, lineName, OBJ_VLINE, 0, t, 0);
   ObjectSetInteger(0, lineName, OBJPROP_COLOR, RegimeColor(category));
   ObjectSetInteger(0, lineName, OBJPROP_STYLE, STYLE_DASHDOT);
   ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, lineName, OBJPROP_BACK, false); // foreground -- see DrawTrend's own comment
   ObjectSetInteger(0, lineName, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, lineName, OBJPROP_TOOLTIP, label);
  }

// Shades the SPAN between one regime transition and the next with a
// flat, low-opacity fill (see DrawTransparentRect) across the chart's
// own current visible price range -- additive to the boundary line
// above (DrawRegimeLine), not a replacement for it. Uses CHART_PRICE_
// MIN/MAX (the currently VISIBLE price axis), same "top to bottom" idea
// as OBJ_VLINE itself, rather than a fixed price band, since a regime
// segment has no price extent of its own -- it's a statement about
// TIME, not price. An earlier round used a directional hatch texture
// here instead of a flat fill; direct user feedback replaced it with
// this simpler wash, matching the S/R/Fib fills' own visual language.
void DrawRegimeRegion(const string objBase, const datetime t1, const datetime t2, const string category, const string label)
  {
   double priceLow  = ChartGetDouble(0, CHART_PRICE_MIN, 0);
   double priceHigh = ChartGetDouble(0, CHART_PRICE_MAX, 0);
   if(priceHigh <= priceLow)
      return; // chart not fully initialized yet (e.g. right after attach) -- skip this refresh, not an error
   string rectName = objBase + "_fill";
   DrawTransparentRect(rectName, priceLow, priceHigh, t1, t2, RegimeColor(category), REGIME_FILL_ALPHA);
   ObjectSetString(0, rectName, OBJPROP_TOOLTIP, label);

   // The canvas bitmap above is a pixel-positioned overlay, not a real
   // chart-native shape -- direct user feedback (twice) that hovering
   // INSIDE the shaded region still showed no tooltip. A genuinely
   // invisible (clrNONE), FILLED OBJ_RECTANGLE spanning the same box
   // gives MT5 a real hit-test AREA covering the whole region (unlike a
   // hollow FILL=false border, whose hit-test is only the thin outline
   // stroke). Deliberately FOREGROUND (BACK=false) here, unlike every
   // other object in this file -- BACK=true objects appear to sit
   // outside MT5's own tooltip hit-test scan entirely (the first attempt
   // used BACK=true and still didn't show a tooltip); since this
   // rectangle is clrNONE either way, making it foreground costs nothing
   // visually -- the canvas bitmap stays the only thing actually drawn
   // on screen.
   string hitName = objBase + "_hit";
   ObjectCreate(0, hitName, OBJ_RECTANGLE, 0, t1, priceHigh, t2, priceLow);
   ObjectSetInteger(0, hitName, OBJPROP_COLOR, clrNONE);
   ObjectSetInteger(0, hitName, OBJPROP_FILL, true);
   ObjectSetInteger(0, hitName, OBJPROP_BACK, false);
   ObjectSetInteger(0, hitName, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, hitName, OBJPROP_TOOLTIP, label);
  }

// Split into a Bands half (drawn in RefreshOverlay's PASS 1, with every
// other fill) and a Tool half (PASS 2, with every other line/border) --
// same reasoning as DrawLevelZoneFill/Border above.
void DrawFiboBands(const string objBase, const double swingHigh, const double swingLow, const bool highIsRecent,
                    const int highSecondsAgo, const int lowSecondsAgo)
  {
   datetime highTime = TimeCurrent() - highSecondsAgo;
   datetime lowTime  = TimeCurrent() - lowSecondsAgo;
   double p0, p1;
   if(highIsRecent) { p0 = swingHigh; p1 = swingLow; }
   else             { p0 = swingLow;  p1 = swingHigh; }

   int levelCount = ArraySize(g_fiboRatios);
   double levelPrices[7];
   for(int i = 0; i < levelCount; i++)
      levelPrices[i] = p0 + (p1 - p0) * g_fiboRatios[i];

   // Fills the BAND between each consecutive pair of levels with a real,
   // low-opacity tint in the theme-adaptive heat-map palette -- direct
   // user request: "fill colors... between the fibonacci level[s]...
   // very low transparency so the candles are clearly visible." Each
   // band's own detail is a hover tooltip only (bands 2-3, 38.2%-61.8%,
   // are the golden zone this account's own setup classifier treats as
   // the real pullback-entry area) -- no permanent on-chart text; see
   // DrawLegendPanel for the one-time color key instead.
   datetime bandT1 = iTime(_Symbol, PERIOD_CURRENT, 40);
   datetime bandT2 = TimeCurrent() + PeriodSeconds(PERIOD_CURRENT) * 10;
   for(int i = 0; i < levelCount - 1; i++)
     {
      double lo = MathMin(levelPrices[i], levelPrices[i + 1]);
      double hi = MathMax(levelPrices[i], levelPrices[i + 1]);
      string bandName = objBase + "_band" + IntegerToString(i);
      string bandLabel = StringFormat("Fib %.1f%%-%.1f%%", g_fiboRatios[i] * 100.0, g_fiboRatios[i + 1] * 100.0);
      if(i == 2 || i == 3)
         bandLabel += " (golden zone)";
      DrawTransparentRect(bandName, lo, hi, bandT1, bandT2, g_fiboLevelColors[i], FIBO_BAND_FILL_ALPHA);
      ObjectSetString(0, bandName, OBJPROP_TOOLTIP, bandLabel);
     }
  }

// MT5's own native Fibonacci retracement tool, anchored at the real
// swing extremes and real times -- point 0 (the "0%" anchor) is
// whichever extreme is more RECENT, matching this account's own
// retracement convention (analysis/chart_structure.py: "retracement is
// measured back from whichever extreme came LAST").
void DrawFiboTool(const string objBase, const double swingHigh, const double swingLow, const bool highIsRecent,
                   const int highSecondsAgo, const int lowSecondsAgo, const string label)
  {
   datetime highTime = TimeCurrent() - highSecondsAgo;
   datetime lowTime  = TimeCurrent() - lowSecondsAgo;
   datetime t0, t1;
   double p0, p1;
   if(highIsRecent)
     {
      t0 = highTime; p0 = swingHigh;
      t1 = lowTime;  p1 = swingLow;
     }
   else
     {
      t0 = lowTime;  p0 = swingLow;
      t1 = highTime; p1 = swingHigh;
     }
   string name = objBase + "_fibo";
   ObjectCreate(0, name, OBJ_FIBO, 0, t0, p0, t1, p1);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, false); // foreground -- see DrawTrend's own comment
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   // The "main" OBJPROP_COLOR on OBJ_FIBO controls its own diagonal
   // 0%-to-100% connector line -- a geometric artifact of the tool (it
   // just joins the two swing-point anchors), not a real trendline, and
   // direct user feedback called it out as "weird... I don't see they
   // follow any pattern." Coloring it identically to the chart's own
   // live background makes it invisible while every individual ratio
   // line (each with its OWN separate OBJPROP_LEVELCOLOR, set below)
   // stays fully visible.
   ObjectSetInteger(0, name, OBJPROP_COLOR, g_chartBackgroundColor);
   // Direct user request 2026-09-05: "make fibonacci line and text all
   // black" -- the per-level tick line AND its accompanying percentage
   // text share ONE OBJPROP_LEVELCOLOR, so both are g_noteColor now
   // (Black on light theme / White on dark, matching this file's own
   // "just needs to read clearly, no signal encoded in this color"
   // convention already used for notes/the legend header) instead of
   // the heat-map palette. DrawFiboBands (the fills between levels)
   // keeps that heat-map palette unchanged -- the golden zone is still
   // visually called out by ITS band's own tint, just not by the tick
   // lines/text anymore.
   int levelCount = ArraySize(g_fiboRatios);
   ObjectSetInteger(0, name, OBJPROP_LEVELS, levelCount);
   for(int i = 0; i < levelCount; i++)
     {
      ObjectSetDouble(0, name, OBJPROP_LEVELVALUE, i, g_fiboRatios[i]);
      ObjectSetInteger(0, name, OBJPROP_LEVELCOLOR, i, g_noteColor);
      ObjectSetInteger(0, name, OBJPROP_LEVELSTYLE, i, STYLE_DOT);
      ObjectSetInteger(0, name, OBJPROP_LEVELWIDTH, i, 1);
      ObjectSetString(0, name, OBJPROP_LEVELTEXT, i, StringFormat("%.1f%%", g_fiboRatios[i] * 100.0));
     }
   ObjectSetString(0, name, OBJPROP_TOOLTIP, label);
  }

void DrawNote(const string name, const int slot, const string text)
  {
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 8);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 18 + slot * 16);
   ObjectSetInteger(0, name, OBJPROP_COLOR, g_noteColor);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
  }

#define LEGEND_PANEL_ALPHA 235
#define LEGEND_FONT_SIZE 18
#define LEGEND_ROW_HEIGHT 30
#define LEGEND_PAD_X 12
#define LEGEND_PAD_Y 12
#define LEGEND_MARGIN_X 4
#define LEGEND_MARGIN_Y 4

// ONE bitmap panel replacing the old per-row OBJ_LABEL stack -- direct
// user request 2026-09-04: "show legend text in a box which should have
// a background fill color so that text does not mix up with the
// background chart." Drawing the background AND every row's text onto a
// SINGLE CCanvas bitmap (Erase for the fill, TextOut per row) means
// there's no separate box-object-vs-text-object alignment to get right,
// and the fill color is g_chartBackgroundColor itself -- the exact
// reference color every row's own text color (the swatch palette from
// DetectTheme) was ALREADY tuned to read clearly against, so contrast is
// guaranteed without inventing a new color decision.
//
// Positioned via CORNER_LEFT_LOWER, but deliberately withOUT relying on
// OBJPROP_ANCHOR for a bitmap object (unlike the old per-row labels,
// which needed ANCHOR_LEFT_LOWER explicitly) -- a bitmap's own default
// reference point is its TOP-LEFT corner, so YDISTANCE is computed as
// (margin + the panel's own real height) here instead, which pins the
// BOTTOM edge at the intended margin from the chart's bottom edge using
// only the well-established, corner-independent XDISTANCE/YDISTANCE
// mechanism every screen-anchored chart object supports.
//
// Called LAST in RefreshOverlay's redraw (direct user request: "show
// legend at the front above all charting elements") -- MT5 stacks
// same-layer (BACK=false) objects by creation order, so being the very
// last thing created this pass is what actually guarantees it renders
// on top of every zone/hatch/line/band drawn earlier in the same pass,
// not just a hopeful ordering.
void DrawLegendPanel()
  {
   // Entry/Stop/Target deliberately NOT keyed here (direct user request
   // 2026-09-04: "remove entry, stop, target labels" from the legend) --
   // each already carries its own hover tooltip (DrawHardLine), and
   // there are at most 3 of them on any one chart at a time, so a
   // dedicated legend row per line was more clutter than clarity.
   string rowText[12];
   color  rowColor[12];
   int n = 0;
   rowText[n] = "Legend";               rowColor[n] = g_noteColor;       n++;
   rowText[n] = "Support zone";         rowColor[n] = g_supportColor;    n++;
   rowText[n] = "Resistance zone";      rowColor[n] = g_resistanceColor; n++;
   rowText[n] = "Realistic move zone";  rowColor[n] = g_atrZoneColor;    n++;
   rowText[n] = "Fibonacci bands";      rowColor[n] = g_fiboColor;       n++;
   rowText[n] = "Golden zone";          rowColor[n] = g_fiboLevelColors[3]; n++;
   rowText[n] = "H1/H4 trendline";      rowColor[n] = g_trendColor;      n++;
   rowText[n] = "Uptrend regime";       rowColor[n] = g_uptrendRegimeColor;   n++;
   rowText[n] = "Downtrend regime";     rowColor[n] = g_downtrendRegimeColor; n++;
   rowText[n] = "Sideways regime";      rowColor[n] = g_sidewaysRegimeColor; n++;
   rowText[n] = "Accumulation";         rowColor[n] = g_accumulationColor;   n++;
   rowText[n] = "Distribution";         rowColor[n] = g_distributionColor;  n++;

   string name = OBJ_PREFIX + "legend_panel";
   ClearLegendCanvas(); // this function's own redraw pass always rebuilds it fresh
   if(!g_legendCanvas.CreateBitmapLabel(name, 0, 0, 10, 10, COLOR_FORMAT_ARGB_NORMALIZE))
      return;
   g_legendCanvasActive = true;
   g_legendCanvas.FontSet("Arial", LEGEND_FONT_SIZE);

   int maxTextWidth = 0;
   for(int i = 0; i < n; i++)
      maxTextWidth = MathMax(maxTextWidth, g_legendCanvas.TextWidth(rowText[i]));

   int panelWidth  = maxTextWidth + LEGEND_PAD_X * 2;
   int panelHeight = n * LEGEND_ROW_HEIGHT + LEGEND_PAD_Y * 2;
   g_legendCanvas.Resize(panelWidth, panelHeight);

   g_legendCanvas.Erase(ToARGB(g_chartBackgroundColor, LEGEND_PANEL_ALPHA));
   for(int i = 0; i < n; i++)
     {
      uint rowArgb = ToARGB(rowColor[i], 255);
      int  rowY    = LEGEND_PAD_Y + i * LEGEND_ROW_HEIGHT;
      if(i == 0)
        {
         // "Legend" header: bold + underline, direct user request. CCanvas
         // text has no bold/underline flag this file trusts without a live
         // test, so both are drawn directly instead: bold via the classic
         // "double-draw offset 1px right" trick (thickens every stroke,
         // works regardless of font/flag support), underline via a real
         // LineHorizontal spanning the header's own measured width, drawn
         // just below its measured height.
         g_legendCanvas.TextOut(LEGEND_PAD_X,     rowY, rowText[i], rowArgb);
         g_legendCanvas.TextOut(LEGEND_PAD_X + 1, rowY, rowText[i], rowArgb);
         int headerWidth  = g_legendCanvas.TextWidth(rowText[i]);
         int headerHeight = g_legendCanvas.TextHeight(rowText[i]);
         g_legendCanvas.LineHorizontal(LEGEND_PAD_X, LEGEND_PAD_X + headerWidth, rowY + headerHeight + 1, rowArgb);
        }
      else
         g_legendCanvas.TextOut(LEGEND_PAD_X, rowY, rowText[i], rowArgb);
     }
   g_legendCanvas.Update();

   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_LOWER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, LEGEND_MARGIN_X);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, LEGEND_MARGIN_Y + panelHeight);
   ObjectSetInteger(0, name, OBJPROP_BACK, false); // foreground -- see this function's own docstring
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
  }

//+------------------------------------------------------------------+
// Called every RefreshSeconds by OnTimer. Reads the file and, if the
// set of lines matching THIS chart's symbol is byte-for-byte identical
// to what was drawn last time, returns immediately WITHOUT touching a
// single object -- direct user feedback: "after few second the region
// flashes/jerk on every refresh." The old behavior unconditionally
// cleared and redrew everything every single poll even when nothing
// about this symbol's own data had changed, which is what produced the
// visible flash. A genuine content change (a new price level, a
// symbol's note/thesis updating, etc.) still triggers a full redraw.
void RefreshOverlay()
  {
   DetectTheme();

   int handle = FileOpen(InputFileName, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
     {
      PrintFormat("Quant2ChartOverlay: FileOpen(%s, FILE_COMMON) failed, error %d -- Python side hasn't written it yet, or it's not in the Common Files folder.",
                  InputFileName, GetLastError());
      return; // not written yet, or nothing to show this run -- not an error
     }

   string matchedRawLines[];
   int matchedCount = 0;
   int totalLines = 0;
   string fingerprint = "";
   while(!FileIsEnding(handle))
     {
      string line = FileReadString(handle);
      if(StringLen(line) == 0)
         continue;
      totalLines++;

      string parts[];
      int n = StringSplit(line, '|', parts);
      if(n < 2 || parts[1] != _Symbol)
         continue;

      ArrayResize(matchedRawLines, matchedCount + 1);
      matchedRawLines[matchedCount] = line;
      matchedCount++;
      fingerprint += line + "\n";
     }
   FileClose(handle);

   static string lastFingerprint = "\x01"; // guarantees a mismatch (a real redraw) on the very first call
   if(fingerprint == lastFingerprint)
      return; // nothing about this symbol changed -- skip the whole clear+redraw, no flicker
   lastFingerprint = fingerprint;

   // Content genuinely changed -- do the full clear + redraw. Canvas
   // specs are rebuilt fresh from this pass (see DrawTransparentRect),
   // so reset the count rather than letting old entries accumulate.
   ClearCanvases(); // must run BEFORE ClearOverlay's generic ObjectDelete sweep -- see its own docstring
   ClearOverlay();
   g_canvasSpecCount = 0;

   // Direct user request 2026-09-04: "show the vertical lines, trend
   // lines and region horizontal boxes over the hatch fill color...
   // line should be at the front." MT5 stacks same-layer objects by
   // creation order, so this now runs in explicit ordered PASSES rather
   // than one combined pass -- every fill first (PASS 1, lowest layer),
   // then every border/trendline/hard-line (PASS 2, on top of ALL
   // fills, not just its own), then the regime boundary vlines last of
   // all "content" objects (on top of PASS 2 too), then DrawLegendPanel
   // (absolute top, foreground). objName is recomputed identically in
   // both passes from the SAME deterministic lineIndex sequence, so a
   // given file line always gets the same object base name regardless
   // of which pass is currently running.
   string regimeObjName[32];
   int    regimeSecAgo[32];
   string regimeCategory[32];
   string regimeLabel[32];
   int    regimeCount = 0;

   // PASS 1 -- fills only (LEVEL/FIBO/ZONE), plus collecting VLINE data
   // (its own fill is drawn separately below, once every segment's real
   // span is known -- see the regime block after PASS 2).
   int lineIndex = 0;
   for(int idx = 0; idx < matchedCount; idx++)
     {
      string parts[];
      int n = StringSplit(matchedRawLines[idx], '|', parts);
      string recType = parts[0];
      string objName = OBJ_PREFIX + IntegerToString(lineIndex);
      lineIndex++;

      if(recType == "LEVEL" && n >= 6)
        {
         double price = StringToDouble(parts[2]);
         double tolerancePct = StringToDouble(parts[5]);
         DrawLevelZoneFill(objName, price, tolerancePct, LevelColor(parts[3]));
        }
      else if(recType == "FIBO" && n >= 7)
        {
         double swingHigh    = StringToDouble(parts[2]);
         double swingLow     = StringToDouble(parts[3]);
         bool   highIsRecent = (parts[4] == "1");
         int    highSecAgo   = (int)StringToInteger(parts[5]);
         int    lowSecAgo    = (int)StringToInteger(parts[6]);
         DrawFiboBands(objName, swingHigh, swingLow, highIsRecent, highSecAgo, lowSecAgo);
        }
      else if(recType == "ZONE" && n >= 5)
        {
         double low  = StringToDouble(parts[2]);
         double high = StringToDouble(parts[3]);
         DrawRegionFill(objName, low, high, g_atrZoneColor);
        }
      else if(recType == "VLINE" && n >= 5 && regimeCount < 32)
        {
         regimeObjName[regimeCount]  = objName;
         regimeSecAgo[regimeCount]   = (int)StringToInteger(parts[2]);
         regimeCategory[regimeCount] = parts[3];
         regimeLabel[regimeCount]    = parts[4];
         regimeCount++;
        }
     }

   // Regime segments arrive in ascending start_index order (oldest
   // first, see ai/chart_overlay.py's own _regime_vline_lines) -- so
   // secAgo DECREASES as i increases. Segment i's own SPAN runs from its
   // own marker to the NEXT segment's marker (that's the whole point of
   // a "transition": category[i] describes everything until the next
   // one begins); the most recent segment has no successor yet, so its
   // span extends to "now" (plus the same small lookahead padding the
   // zone/fib rectangles already use). Fills only here -- the boundary
   // vlines themselves are drawn AFTER pass 2, see below.
   if(regimeCount > 0)
     {
      datetime nowT = TimeCurrent();
      for(int i = 0; i < regimeCount; i++)
        {
         datetime t1 = nowT - regimeSecAgo[i];
         datetime t2 = (i + 1 < regimeCount) ? (nowT - regimeSecAgo[i + 1]) : (nowT + PeriodSeconds(PERIOD_CURRENT) * 10);
         DrawRegimeRegion(regimeObjName[i], t1, t2, regimeCategory[i], regimeLabel[i]);
        }
     }

   // PASS 2 -- everything that's a line/border/tool rather than a fill:
   // created AFTER every fill above, so it renders on top of all of
   // them. VLINE is deliberately absent here (handled once, after this
   // loop, using the SAME regime* arrays PASS 1 already collected).
   int noteSlot = 0;
   string briefing = "";
   lineIndex = 0;
   for(int idx = 0; idx < matchedCount; idx++)
     {
      string parts[];
      int n = StringSplit(matchedRawLines[idx], '|', parts);
      string recType = parts[0];
      string objName = OBJ_PREFIX + IntegerToString(lineIndex);
      lineIndex++;

      if(recType == "LEVEL" && n >= 6)
        {
         double price = StringToDouble(parts[2]);
         double tolerancePct = StringToDouble(parts[5]);
         DrawLevelZoneBorder(objName, price, tolerancePct, LevelColor(parts[3]), parts[4]);
        }
      else if(recType == "FIBO" && n >= 7)
        {
         double swingHigh    = StringToDouble(parts[2]);
         double swingLow     = StringToDouble(parts[3]);
         bool   highIsRecent = (parts[4] == "1");
         int    highSecAgo   = (int)StringToInteger(parts[5]);
         int    lowSecAgo    = (int)StringToInteger(parts[6]);
         string label        = (n >= 8) ? parts[7] : "Fibonacci retracement";
         DrawFiboTool(objName, swingHigh, swingLow, highIsRecent, highSecAgo, lowSecAgo, label);
        }
      else if(recType == "ZONE" && n >= 5)
        {
         double low  = StringToDouble(parts[2]);
         double high = StringToDouble(parts[3]);
         DrawRegionBorder(objName, low, high, g_atrZoneColor, parts[4]);
        }
      else if(recType == "TREND" && n >= 6)
        {
         double price1 = StringToDouble(parts[2]);
         double price2 = StringToDouble(parts[3]);
         int    secBack = (int)StringToInteger(parts[4]);
         DrawTrend(objName, price1, price2, secBack, parts[5], (n >= 7) ? parts[6] : "");
        }
      else if(recType == "ENTRY" && n >= 4)
        {
         double price = StringToDouble(parts[2]);
         DrawHardLine(objName, price, g_entryColor, "Entry (" + parts[3] + ")");
        }
      else if(recType == "STOP" && n >= 3)
        {
         double price = StringToDouble(parts[2]);
         DrawHardLine(objName, price, g_stopColor, "Stop");
        }
      else if(recType == "TARGET" && n >= 3)
        {
         double price = StringToDouble(parts[2]);
         DrawHardLine(objName, price, g_targetColor, "Target");
        }
      else if(recType == "NOTE" && n >= 3)
        {
         string text = parts[2];
         for(int i = 3; i < n; i++)
            text = text + "|" + parts[i];
         DrawNote(OBJ_PREFIX + "note_" + IntegerToString(noteSlot), noteSlot, text);
         noteSlot++;
        }
      else if(recType == "LOG" && n >= 3)
        {
         string text = parts[2];
         for(int i = 3; i < n; i++)
            text = text + "|" + parts[i];
         briefing = text;
        }
     }

   // Regime boundary vlines: drawn LAST of all "content" objects (after
   // every border/trendline/hard-line in PASS 2 too), so they're the
   // frontmost thing short of the legend panel itself.
   for(int i = 0; i < regimeCount; i++)
      DrawRegimeLine(regimeObjName[i], regimeSecAgo[i], regimeCategory[i], regimeLabel[i]);

   // Drawn LAST -- see DrawLegendPanel's own docstring for why creation
   // order here is what actually guarantees it renders on top of
   // everything else this pass, not just a hopeful ordering.
   DrawLegendPanel();

   ChartRedraw(0);

   // Reaching here already means the content fingerprint changed, so
   // this always reflects a genuine update, not a repeated poll.
   PrintFormat("Quant2ChartOverlay: %s -- %d total line(s) in file, %d matched this symbol.",
               _Symbol, totalLines, matchedCount);

   // The actual strategic update -- this account's own real, already-
   // computed thesis (side, entry/stop/target, the reasoning text),
   // not a bare object/line count. Direct user request 2026-09-04:
   // "give more strategic updates like a commander giving instructions
   // ... not just post health status logs."
   if(briefing != "")
      PrintFormat("Quant2ChartOverlay [%s] STRATEGY BRIEFING: %s", _Symbol, briefing);
  }
//+------------------------------------------------------------------+
