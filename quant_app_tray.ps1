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

$pythonExe = "C:\Users\mubas\AppData\Local\Programs\Python\Python314\python.exe"
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
    $cmdLine = "`"$pythonExe`" -m streamlit run app.py --server.headless true >> `"$logFile`" 2>&1"
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
