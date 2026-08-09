//+------------------------------------------------------------------+
//| DumpSymbolSpecs.mq5                                               |
//|                                                                    |
//| Dumps every symbol in the terminal's own symbol database (not just |
//| the ones currently visible in Market Watch) to a CSV file, using   |
//| MQL5's native SymbolInfoDouble/SymbolInfoString calls running      |
//| *inside* the terminal process itself.                              |
//|                                                                    |
//| This exists as a fallback data path for Quant2's Python side       |
//| (see data/mt5_source.py:_load_symbol_spec_from_csv), which reads   |
//| the terminal through MetaTrader5's external Python API instead.    |
//| That external API has a documented quirk where a symbol can show   |
//| as "visible" while still not being "selected" for the calling      |
//| API session's own internal cache, which symbol_info() depends on   |
//| -- occasionally causing real spec data to read back as None even   |
//| though the terminal itself has it (confirmed live for SP500-SE26). |
//| A script running natively inside the terminal has no such session  |
//| boundary to cross, so it can't hit that particular failure mode.   |
//|                                                                    |
//| Run manually from the Navigator's Scripts list whenever you want   |
//| a fresh dump (e.g. after adding new symbols to Market Watch, or    |
//| periodically via a chart timer if you turn this into an EA later). |
//| Output goes to <Data Folder>\MQL5\Files\symbol_specs.csv -- find   |
//| the Data Folder via MetaEditor or the terminal's File menu.        |
//+------------------------------------------------------------------+
#property copyright "Quant2"
#property script_show_inputs

input string OutputFileName = "symbol_specs.csv";
// Default true: SYMBOL_MARGIN_INITIAL reads back 0 for any symbol the
// broker isn't actively quoting right now (confirmed live: every expired
// 2021-2025 contract-month dumped as 0 margin, while every live 2026
// symbol came back correct) -- dumping the full historical symbol tree
// (thousands of rows) just buries the ~80-100 rows that actually matter
// under noise that looks like a bug but isn't one. Set false only if you
// specifically need a symbol that isn't currently in Market Watch.
input bool   MarketWatchOnly = true;

//+------------------------------------------------------------------+
void OnStart()
  {
   int total = SymbolsTotal(MarketWatchOnly);
   if(total <= 0)
     {
      Print("DumpSymbolSpecs: no symbols found (MarketWatchOnly=", MarketWatchOnly, ")");
      return;
     }

   int handle = FileOpen(OutputFileName, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
   if(handle == INVALID_HANDLE)
     {
      Print("DumpSymbolSpecs: failed to open ", OutputFileName, ", error ", GetLastError());
      return;
     }

   FileWrite(handle, "symbol", "volume_min", "volume_step", "volume_max",
             "trade_contract_size", "currency_margin", "margin_initial");

   int written = 0;
   for(int i = 0; i < total; i++)
     {
      string symbol = SymbolName(i, MarketWatchOnly);
      if(symbol == "")
         continue;

      // SymbolSelect ensures this symbol's data is loaded even if it
      // isn't currently in Market Watch -- mirrors why the Python side
      // needs symbol_select() too, just with no cross-process session
      // boundary to fail across here.
      SymbolSelect(symbol, true);

      double volume_min      = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double volume_step     = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
      double volume_max      = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      double contract_size   = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      string currency_margin = SymbolInfoString(symbol, SYMBOL_CURRENCY_MARGIN);
      double margin_initial  = SymbolInfoDouble(symbol, SYMBOL_MARGIN_INITIAL);

      FileWrite(handle, symbol, volume_min, volume_step, volume_max,
                contract_size, currency_margin, margin_initial);
      written++;
     }

   FileClose(handle);
   Print("DumpSymbolSpecs: wrote ", written, " of ", total, " symbol(s) to ", OutputFileName);
  }
//+------------------------------------------------------------------+
