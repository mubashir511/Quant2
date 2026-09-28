# Quant2 Webapp launcher with a system-tray presence.
#
# Real user request this satisfies: (1) show a visible confirmation that
# the app actually started, (2) then get out of the way into the
# notification-area tray (like Task Manager's own "minimize to tray"
# option) instead of sitting on the taskbar as a bare console window.
#
# Launched by quant_app.bat, which is in turn what the desktop shortcut
# ("Quant2 Webapp.lnk") and the "Quant2Webapp" Scheduled Task both point
# at — so this same tray behavior applies whether the app is started
# manually or automatically at logon. See project_247_deployment memory
# for that existing 24/7 setup this builds on top of, unchanged.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
$logFile = Join-Path $root "streamlit_log.txt"
$icoPath = Join-Path $root "quant_app.ico"
$appUrl = "http://localhost:8501"
$taskName = "Quant2Webapp"

# --- Win32 interop: hiding the console window (not closing it — the
# process stays alive, running the tray icon's own message loop) has no
# built-in PowerShell cmdlet, so this needs a couple of raw user32/
# kernel32 calls, same general technique already used in this project for
# building quant_app.ico itself (System.Drawing via Add-Type). ---
Add-Type -Namespace Quant2Native -Name Window -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
'@
$SW_HIDE = 0

function Test-PortOpen {
    param([string]$ComputerName, [int]$Port, [int]$TimeoutMs = 300)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $client.BeginConnect($ComputerName, $Port, $null, $null)
        $connected = $asyncResult.AsyncWaitHandle.WaitOne($TimeoutMs)
        if ($connected -and $client.Connected) {
            $client.Close()
            return $true
        }
        $client.Close()
        return $false
    } catch {
        return $false
    }
}

# --- Already running? (e.g. the Scheduled Task already started it at
# logon, and this is a manual double-click of the shortcut on top of
# that) — don't start a second MT5-connected instance; MT5's own Python
# package supports exactly one live connection per process, and running
# two independent processes against the same live account is a real risk
# worth avoiding outright, not just a wasted-resources annoyance. ---
if (Test-PortOpen -ComputerName "localhost" -Port 8501) {
    Write-Host "Quant2 Webapp is already running at $appUrl"
    Write-Host "Opening it in your browser..."
    Start-Process $appUrl
    Start-Sleep -Seconds 3
    exit 0
}

# Real bug found live: async output capture via Register-ObjectEvent on
# Process.OutputDataReceived (an earlier version of this function) never
# actually fired in this exact invocation mode (powershell.exe -STA
# -File ...) — confirmed with an isolated repro where a child process's
# stdout was silently lost even though the child genuinely produced it.
# Went back to the same reliable mechanism the old quant_app.bat always
# used instead: a real shell (`cmd.exe /c "... >> log 2>&1"`) doing the
# append-mode redirection itself, no .NET-side event plumbing needed.
# `cmd /c` waits for its whole redirected command line to finish before
# it exits, so the tracked process's own HasExited still accurately
# reflects whether Streamlit itself is still running.
function Start-Streamlit {
    # --server.address 0.0.0.0 (added for remote/mobile access via
    # Tailscale, direct user request): binds to every network interface
    # on this PC, not just localhost, so a request arriving over the
    # Tailscale virtual adapter (its own private 100.64.0.0/10 network)
    # can actually reach it. Safe specifically BECAUSE the Windows
    # Firewall rule set up alongside this only allows inbound port 8501
    # from that same Tailscale range — binding wide is what lets that
    # firewall rule be the actual access boundary instead of Streamlit's
    # own bind address; the app.py password gate is the second,
    # independent layer in case either of those is ever misconfigured.
    $cmdLine = "`"$pythonExe`" -m streamlit run app.py --server.headless true --server.address 0.0.0.0 >> `"$logFile`" 2>&1"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "cmd.exe"
    # The extra outer quote pair matters: cmd.exe /c has a documented
    # quirk where, if the command immediately following /c itself starts
    # with a quote (ours does — the quoted python.exe path), its own
    # quote-stripping rules corrupt the rest of the parse and it silently
    # exits nonzero without ever actually running anything. Confirmed
    # live: without this, cmd.exe exited in ~1s (code 1) with NOTHING
    # written to the log at all — not even an error — while a command
    # that doesn't start with a quote (e.g. a bare "echo ...") worked
    # fine, isolating this exact quoting rule as the cause. Wrapping the
    # whole thing in one more quote pair is the standard, documented fix.
    $psi.Arguments = "/c `"$cmdLine`""
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.EnableRaisingEvents = $true
    $proc.Start() | Out-Null
    return $proc
}

# Process.Kill() only kills the exact tracked process (the cmd.exe
# wrapper above), not its python.exe/streamlit descendant — killing just
# the wrapper would leave Streamlit itself running as an orphan, quietly
# defeating both "Exit stops the app" and "Restart" alike. taskkill's
# /T flag kills the whole process tree, which is what's actually needed.
function Stop-ProcessTree {
    param([int]$ProcessId)
    try {
        Start-Process -FilePath "taskkill.exe" -ArgumentList "/PID $ProcessId /T /F" -WindowStyle Hidden -Wait -ErrorAction SilentlyContinue | Out-Null
    } catch {}
}

Write-Host "Starting Quant2 Webapp..."
$script:intentionalStop = $false
$script:proc = Start-Streamlit

# --- Wait for a real, external confirmation the server is actually
# accepting connections (a TCP probe) rather than just "the process
# hasn't crashed yet" — matches this session's own "verify with real
# evidence, not an assumption" standard elsewhere in this project. ---
$readyTimeoutSec = 45
$waited = 0.0
$isUp = $false
while ($waited -lt $readyTimeoutSec) {
    if ($script:proc.HasExited) { break }
    if (Test-PortOpen -ComputerName "localhost" -Port 8501) { $isUp = $true; break }
    Start-Sleep -Milliseconds 500
    $waited += 0.5
}

if (-not $isUp) {
    Write-Host ""
    Write-Host "Quant2 Webapp did NOT start within $readyTimeoutSec seconds."
    Write-Host "Leaving this window open — see streamlit_log.txt for details."
    Write-Host ""
    if (Test-Path $logFile) {
        Write-Host "--- Last lines of streamlit_log.txt ---"
        Get-Content $logFile -Tail 25
    }
    Write-Host ""
    Read-Host "Press Enter to close this window (the app will stay stopped)"
    $script:intentionalStop = $true
    if (-not $script:proc.HasExited) { Stop-ProcessTree -ProcessId $script:proc.Id }
    exit 1
}

Write-Host ""
Write-Host "Quant2 Webapp is running: $appUrl"
Write-Host "Setting up the tray icon..."

function Stop-Quant2 {
    $script:intentionalStop = $true
    if (-not $script:proc.HasExited) { Stop-ProcessTree -ProcessId $script:proc.Id }
    # Best-effort: if this instance was started by the Scheduled Task
    # (e.g. the AtLogOn trigger), tell Task Scheduler to stop it
    # explicitly too — otherwise its own RestartCount/RestartInterval
    # crash-recovery policy could relaunch the app a minute after the
    # user deliberately exited it, silently undoing "Exit stops the
    # app". Harmless no-op if this instance wasn't started that way
    # (e.g. a manual double-click of the shortcut).
    try { Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue } catch {}
}

# --- Build the ENTIRE tray icon (and confirm it actually worked) BEFORE
# hiding the console — real bug found live: a genuine, reproducible
# timing race in Windows PowerShell's Add-Type -AssemblyName
# System.Windows.Forms occasionally leaves a type not fully usable right
# away (confirmed by reproducing it repeatedly: sometimes ContextMenuStrip
# failed to resolve, sometimes NotifyIcon itself came back unusable —
# different type each time, same root cause). Building this while the
# console is still visible, retrying a few times, and only hiding the
# console once it's confirmed working means a failure here can never
# again make the whole window silently vanish — it now falls back to
# just leaving the console open instead. ---
$trayReady = $false
$maxAttempts = 4
for ($attempt = 1; $attempt -le $maxAttempts -and -not $trayReady; $attempt++) {
    try {
        if ($attempt -gt 1) {
            Write-Host "Tray icon setup didn't take — retrying (attempt $attempt of $maxAttempts)..."
            Start-Sleep -Milliseconds 800
        }
        if ($script:notifyIcon) {
            try { $script:notifyIcon.Visible = $false; $script:notifyIcon.Dispose() } catch {}
            $script:notifyIcon = $null
        }

        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        # Forces the assembly to actually finish initializing before the
        # types below are used — the concrete, documented mitigation for
        # the race described above.
        [System.Windows.Forms.Application]::EnableVisualStyles()

        try {
            $trayIconImg = New-Object System.Drawing.Icon($icoPath)
        } catch {
            $trayIconImg = [System.Drawing.SystemIcons]::Application
        }

        $script:notifyIcon = New-Object System.Windows.Forms.NotifyIcon
        $script:notifyIcon.Icon = $trayIconImg
        $script:notifyIcon.Text = "Quant2 Webapp - running"

        $menu = New-Object System.Windows.Forms.ContextMenuStrip

        $openItem = $menu.Items.Add("Open Quant2 in browser")
        $openItem.add_Click({ Start-Process $appUrl })

        $logItem = $menu.Items.Add("View log")
        $logItem.add_Click({ Start-Process notepad.exe $logFile })

        $restartItem = $menu.Items.Add("Restart app")
        $restartItem.add_Click({
            $script:notifyIcon.ShowBalloonTip(2000, "Quant2 Webapp", "Restarting...", [System.Windows.Forms.ToolTipIcon]::Info)
            $script:intentionalStop = $true
            if (-not $script:proc.HasExited) { Stop-ProcessTree -ProcessId $script:proc.Id }
            Start-Sleep -Seconds 1
            $script:proc = Start-Streamlit
            $script:intentionalStop = $false
        })

        $menu.Items.Add("-") | Out-Null

        $exitItem = $menu.Items.Add("Exit (stops the app)")
        $exitItem.add_Click({
            Stop-Quant2
            $script:notifyIcon.Visible = $false
            [System.Windows.Forms.Application]::Exit()
        })

        $script:notifyIcon.ContextMenuStrip = $menu
        $script:notifyIcon.add_DoubleClick({ Start-Process $appUrl })
        $script:notifyIcon.Visible = $true

        $trayReady = $true
    } catch {
        Write-Host "Tray icon setup attempt $attempt failed: $_"
    }
}

if ($trayReady) {
    Write-Host "Minimizing to the system tray..."
    Start-Sleep -Seconds 2

    # --- Hide the console; the tray icon takes over as the only visible
    # presence from here on. ---
    $hwnd = [Quant2Native.Window]::GetConsoleWindow()
    [Quant2Native.Window]::ShowWindow($hwnd, $SW_HIDE) | Out-Null

    $script:notifyIcon.ShowBalloonTip(4000, "Quant2 Webapp", "Running at $appUrl`nRight-click the tray icon for options.", [System.Windows.Forms.ToolTipIcon]::Info)

    # If Streamlit itself crashes while idling in the tray (not a
    # deliberate Restart/Exit from the menu above), exit this wrapper too
    # so the Scheduled Task's own crash-auto-restart safety net (see
    # project_247_deployment memory: RestartCount=3/RestartInterval=1min)
    # actually gets to fire — a tray icon quietly surviving a dead app
    # behind it would otherwise defeat that guarantee.
    Register-ObjectEvent -InputObject $script:proc -EventName Exited -Action {
        if (-not $script:intentionalStop) {
            $script:notifyIcon.Visible = $false
            [System.Windows.Forms.Application]::Exit()
        }
    } | Out-Null

    # --- Clerk's own always-on scheduler (direct user request, 2026-09-
    # 19): clerk_execution_job.py is already a complete, standalone,
    # no-UI-dependency script (no Streamlit, no password gate) that only
    # actually does anything when ai.clerk_execution.is_execution_due
    # says so, behind the same cross-process job_lock.py lock the in-app
    # trigger already uses — this just gives it a NEW trigger that lives
    # inside the tray's own always-pumping message loop, independent of
    # whether any browser tab is open or logged in. Real incident this
    # closes, not hypothetical: Clerk went silent for up to 153 real
    # minutes in one stretch, with no app restart and no Windows sleep
    # event in that window (both confirmed directly) — simply because no
    # browser tab was actively connected to drive the OLD, sole trigger
    # (a Streamlit st.fragment(run_every=...) in app.py, which also sits
    # AFTER that file's own password gate — so the old trigger needed a
    # logged-in tab, not just an open one). 1 minute is deliberately
    # finer-grained than any realistic configured Clerk interval;
    # clerk_execution_job.py's own docstring already establishes this
    # exact "poll often, let the script itself decide if it's due"
    # principle, and it exits almost immediately with no MT5 connection
    # at all on every poll that isn't due — this costs nothing extra on
    # the polls that don't matter. Fire-and-forget by construction (a
    # freshly-started child process, never awaited) so however long a
    # real MT5+local-LLM check takes underneath (up to config.CLERK_
    # EXECUTION_RUN_TIMEOUT_SECONDS, 25 minutes) never blocks this
    # timer's own tick, the tray icon's own responsiveness, or Streamlit
    # itself — same ProcessStartInfo/UseShellExecute=false/CreateNoWindow
    # shape as Start-Streamlit above, for the same reasons.
    $script:clerkTimer = New-Object System.Windows.Forms.Timer
    $script:clerkTimer.Interval = 60000
    $script:clerkTimer.Add_Tick({
        try {
            $clerkPsi = New-Object System.Diagnostics.ProcessStartInfo
            $clerkPsi.FileName = $pythonExe
            $clerkPsi.Arguments = "clerk_execution_job.py"
            $clerkPsi.WorkingDirectory = $root
            $clerkPsi.UseShellExecute = $false
            $clerkPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($clerkPsi) | Out-Null
        } catch {}
    })
    $script:clerkTimer.Start()

    # --- The Sentinel (2026-09-25, plan W5): a one-minute, LLM-free profit-trail ratchet. clerk_sentinel_job.py exits
    # immediately (no MT5 connection) unless the Sentinel and the Clerk are both enabled, no mega session is running, a
    # filled position is tracked and no Clerk poll holds the execution lock; it is log-only until SENTINEL_LOG_ONLY=0.
    # Same fire-and-forget ProcessStartInfo shape as the Clerk timer above.
    $script:sentinelTimer = New-Object System.Windows.Forms.Timer
    $script:sentinelTimer.Interval = 60000
    $script:sentinelTimer.Add_Tick({
        try {
            $sentinelPsi = New-Object System.Diagnostics.ProcessStartInfo
            $sentinelPsi.FileName = $pythonExe
            $sentinelPsi.Arguments = "clerk_sentinel_job.py"
            $sentinelPsi.WorkingDirectory = $root
            $sentinelPsi.UseShellExecute = $false
            $sentinelPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($sentinelPsi) | Out-Null
        } catch {}
    })
    $script:sentinelTimer.Start()

    # --- The Clerk FAST LANE (2026-09-26): the deterministic Clerk pipeline every minute with NO model call, so a failed Mega
    # entry, a structured trigger or a stop ratchet does not wait a whole review interval plus the local-model round (a full poll
    # was 5-11 minutes apart in practice: one candle). clerk_fast_job.py exits before touching MT5 unless there is work, the full poll
    # is not due, and no other process holds the execution lock. Same fire-and-forget shape as the timers above.
    $script:fastLaneTimer = New-Object System.Windows.Forms.Timer
    $script:fastLaneTimer.Interval = 60000
    $script:fastLaneTimer.Add_Tick({
        try {
            $fastPsi = New-Object System.Diagnostics.ProcessStartInfo
            $fastPsi.FileName = $pythonExe
            $fastPsi.Arguments = "clerk_fast_job.py"
            $fastPsi.WorkingDirectory = $root
            $fastPsi.UseShellExecute = $false
            $fastPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($fastPsi) | Out-Null
        } catch {}
    })
    $script:fastLaneTimer.Start()

    # --- The Clerk THINKING job (2026-09-26): the slow, local-model half of the Clerk, run apart from the acting half so a
    # minutes-long model round can never hold up an order. clerk_think_job.py does real work only when a thinking pass is due, under
    # its OWN lock; the fast lane and the regular poll use the verdicts it stores. Same fire-and-forget shape as the timers above.
    $script:thinkTimer = New-Object System.Windows.Forms.Timer
    $script:thinkTimer.Interval = 60000
    $script:thinkTimer.Add_Tick({
        try {
            $thinkPsi = New-Object System.Diagnostics.ProcessStartInfo
            $thinkPsi.FileName = $pythonExe
            $thinkPsi.Arguments = "clerk_think_job.py"
            $thinkPsi.WorkingDirectory = $root
            $thinkPsi.UseShellExecute = $false
            $thinkPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($thinkPsi) | Out-Null
        } catch {}
    })
    $script:thinkTimer.Start()

    # --- Continuation Watch (2026-09-28, phase 2): after a WINNING close, checks whether the pattern kept running and,
    # if so, asks a model whether a fresh continuation trade is worth it — LOG-ONLY today (no order is sent). Its own
    # lock, separate from the Clerk's; exits at once with no MT5 call when nothing is being watched. Same fire-and-forget
    # shape as the timers above.
    $script:continuationWatchTimer = New-Object System.Windows.Forms.Timer
    $script:continuationWatchTimer.Interval = 60000
    $script:continuationWatchTimer.Add_Tick({
        try {
            $cwPsi = New-Object System.Diagnostics.ProcessStartInfo
            $cwPsi.FileName = $pythonExe
            $cwPsi.Arguments = "continuation_watch_job.py"
            $cwPsi.WorkingDirectory = $root
            $cwPsi.UseShellExecute = $false
            $cwPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($cwPsi) | Out-Null
        } catch {}
    })
    $script:continuationWatchTimer.Start()

    # --- Researcher's own always-on scheduler (direct user request,
    # "free researcher as well", 2026-09-19, same day and same root cause
    # as Clerk's fix above): researcher_job.py is the exact same shape as
    # clerk_execution_job.py — standalone, no Streamlit/password
    # dependency, already gated behind ai.researcher.is_researcher_due and
    # the same cross-process job_lock.py lock family (its own lock file,
    # so it can never collide with Clerk's). The one real difference is
    # cadence: Researcher only actually does anything once per day, in a
    # config.RESEARCHER_GRACE_MINUTES-wide window around config.
    # RESEARCHER_TRIGGER_HOUR_UTC:RESEARCHER_TRIGGER_MINUTE_UTC (~12:30
    # UTC) rather than continuously, so a 5-minute tray-side poll is
    # already far finer-grained than that window needs — no reason to
    # spawn a python.exe every single minute just to have it immediately
    # exit "not due yet" for the other ~23 hours of the day. Same fire-
    # and-forget ProcessStartInfo shape as the Clerk timer above.
    $script:researcherTimer = New-Object System.Windows.Forms.Timer
    $script:researcherTimer.Interval = 300000
    $script:researcherTimer.Add_Tick({
        try {
            $researcherPsi = New-Object System.Diagnostics.ProcessStartInfo
            $researcherPsi.FileName = $pythonExe
            $researcherPsi.Arguments = "researcher_job.py"
            $researcherPsi.WorkingDirectory = $root
            $researcherPsi.UseShellExecute = $false
            $researcherPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($researcherPsi) | Out-Null
        } catch {}
    })
    $script:researcherTimer.Start()

    # --- Mega Session's own always-on scheduler (direct user request,
    # "now free the mega session as well", 2026-09-19, same fix as Clerk
    # and Researcher above): mega_analysis_job.py is the same standalone
    # shape as the other two jobs, and its own module docstring already
    # documents an intended 15-minute OS-level poll cadence (it was
    # originally written for a Windows Scheduled Task, "Quant2MegaAnalysis",
    # that in fact was never actually configured on this machine — same
    # gap as Clerk and Researcher had) — 15 minutes here just finally
    # supplies that already-designed-for cadence. The actual analysis
    # still only runs once/day (is_due() gates that, exactly like
    # Researcher's is_researcher_due()), so most ticks no-op immediately.
    # Same fire-and-forget ProcessStartInfo shape as the other two timers.
    #
    # Enable/disable safety (direct user request, same message as this
    # timer): mega_analysis_job.py's own main() already calls
    # read_mega_analysis_enabled() as the VERY FIRST check, before even
    # is_due() or the lock — confirmed by reading the file directly, not
    # assumed. That reads the exact same config.MEGA_ANALYSIS_ENABLED_FILE
    # that app.py's UI toggle writes via set_mega_analysis_enabled(), so
    # switching the toggle off in the dashboard makes this timer's own
    # spawned process exit immediately without touching MT5, the lock, or
    # anything else — this timer adds a trigger, it does not and cannot
    # bypass the toggle. Clerk's existing timer above has the identical
    # property: clerk_execution_job.py checks read_clerk_execution_enabled()
    # first (and ai.clerk_execution.run_clerk_execution_check() checks it
    # AGAIN internally as a second, belt-and-braces gate) against
    # config.CLERK_EXECUTION_ENABLED_FILE, the same file app.py's Clerk
    # toggle writes — both re-verified directly while adding this timer,
    # specifically because the user asked for an explicit guarantee that
    # a UI-disabled session can never keep running in the background.
    $script:megaTimer = New-Object System.Windows.Forms.Timer
    $script:megaTimer.Interval = 900000
    $script:megaTimer.Add_Tick({
        try {
            $megaPsi = New-Object System.Diagnostics.ProcessStartInfo
            $megaPsi.FileName = $pythonExe
            $megaPsi.Arguments = "mega_analysis_job.py"
            $megaPsi.WorkingDirectory = $root
            $megaPsi.UseShellExecute = $false
            $megaPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($megaPsi) | Out-Null
        } catch {}
    })
    $script:megaTimer.Start()

    # --- Trade Audit's own always-on scheduler (direct user request,
    # "retrospectively audit every closed trade once a day after the
    # real US/FTMO forex-session close", 2026-09-20): trade_audit_job.py
    # is the same standalone shape as the other three jobs — no
    # Streamlit/password dependency, gated behind ai.trade_audit.
    # is_trade_audit_due and its own job_lock.py lock file, so it can
    # never collide with any of the other three. DELIBERATE DIFFERENCE
    # from the other three timers' own trigger reasoning: Clerk/
    # Researcher/Mega all anchor to a fixed UTC hour/minute (accepted
    # DST drift, documented on each of their own config constants)
    # because they only need to land somewhere roughly "before/after the
    # US day" — this job instead needs to track one real, precise
    # external event (5:00 PM America/New_York, the real FTMO/US-forex-
    # session daily close) as exactly as possible, so ai.trade_audit.py
    # computes that instant FRESH EACH DAY via Python's stdlib zoneinfo
    # (correctly DST-aware: 21:00 UTC during EDT, 22:00 UTC during EST)
    # rather than reading a fixed HOUR_UTC/MINUTE_UTC pair — see
    # config.py's own TRADE_AUDIT_TRIGGER_HOUR_LOCAL comment for the
    # full reasoning. 5 minutes here is already far finer-grained than
    # this job's own ~2-hour due-window needs (a config.TRADE_AUDIT_
    # SETTLEMENT_BUFFER_MINUTES buffer after the real close, then a
    # config.TRADE_AUDIT_GRACE_MINUTES-wide window) — same "no reason to
    # spawn python.exe every single minute" reasoning as the Researcher
    # timer above. Same fire-and-forget ProcessStartInfo shape as the
    # other three timers.
    #
    # Zero-API-calls-when-idle (direct user request): ai.trade_audit.
    # run_trade_audit_check's own FIRST action, before touching any
    # model, is ai.trade_journal.find_unaudited_closed_stories() — a
    # cheap local directory scan — confirmed by reading ai/trade_audit.py
    # directly.
    $script:tradeAuditTimer = New-Object System.Windows.Forms.Timer
    $script:tradeAuditTimer.Interval = 300000
    $script:tradeAuditTimer.Add_Tick({
        try {
            $tradeAuditPsi = New-Object System.Diagnostics.ProcessStartInfo
            $tradeAuditPsi.FileName = $pythonExe
            $tradeAuditPsi.Arguments = "trade_audit_job.py"
            $tradeAuditPsi.WorkingDirectory = $root
            $tradeAuditPsi.UseShellExecute = $false
            $tradeAuditPsi.CreateNoWindow = $true
            [System.Diagnostics.Process]::Start($tradeAuditPsi) | Out-Null
        } catch {}
    })
    $script:tradeAuditTimer.Start()

    [System.Windows.Forms.Application]::Run()

    if (-not $script:intentionalStop) {
        # Reached only via the crash path above — nonzero exit so this is
        # legible as a real failure (not just an ordinary intentional
        # close) if anyone ever inspects the process's own exit code.
        exit 1
    }
    exit 0
}

# --- Tray icon setup never succeeded after retries. Console stays
# visible (never hidden) so this is never a silent failure — degrades to
# exactly the pre-tray-icon behavior: Quant2 keeps running normally,
# this window just stays open instead of minimizing. ---
Write-Host ""
Write-Host "Couldn't set up the tray icon after $maxAttempts attempts — Quant2 Webapp is still"
Write-Host "running normally at $appUrl, this window will just stay open instead of minimizing."
Write-Host "Close this window (or press Ctrl+C) to stop the app."
while (-not $script:proc.HasExited) {
    Start-Sleep -Seconds 2
}
exit 1
