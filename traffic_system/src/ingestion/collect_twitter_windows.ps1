# Fallback collector for environments where the Python virtualenv is unavailable.
# Run from traffic_system: powershell -ExecutionPolicy Bypass -File src/ingestion/collect_twitter_windows.ps1
$ErrorActionPreference = 'Stop'

$key = (Get-Content .env | Where-Object { $_ -match '^TWITTER_API_KEY=' } |
    ForEach-Object { $_.Split('=', 2)[1] })
if ([string]::IsNullOrWhiteSpace($key)) { throw 'TWITTER_API_KEY is not configured in .env.' }

$dates = @('2026-05-04', '2026-05-06', '2026-05-08', '2026-05-11', '2026-05-13',
    '2026-05-15', '2026-05-18', '2026-05-20', '2026-05-22', '2026-05-25')
$hours = @(@(7, 9), @(17, 19))
$root = Join-Path (Get-Location) 'data/raw/twitter'
$totalPosts = 0
$totalErrors = 0

foreach ($date in $dates) {
    foreach ($hour in $hours) {
        $start = [DateTimeOffset]::Parse("$date" + 'T' + ('{0:D2}' -f $hour[0]) + ':00:00+08:00')
        $end = [DateTimeOffset]::Parse("$date" + 'T' + ('{0:D2}' -f $hour[1]) + ':00:00+08:00')
        $posts = [System.Collections.Generic.List[object]]::new()
        $seen = [System.Collections.Generic.HashSet[string]]::new()
        $slices = [System.Collections.Generic.List[object]]::new()

        for ($cursor = $start; $cursor -lt $end; $cursor = $cursor.AddMinutes(5)) {
            $sliceEnd = $cursor.AddMinutes(5)
            if ($sliceEnd -gt $end) { $sliceEnd = $end }
            $query = '((traffic OR trapik OR "traffic jam" OR "bumper to bumper" OR "heavy traffic" OR aksidente OR accident OR baha OR flooding OR "road closure" OR sunog OR breakdown) (EDSA OR "Roxas Boulevard" OR Makati OR Pasay OR "Quezon City" OR Manila) OR (#MMDAAlert OR #MetroManila OR #TrafficUpdate OR #EDSATraffic OR "traffic advisory")) since_time:' + $cursor.ToUnixTimeSeconds() + ' until_time:' + $sliceEnd.ToUnixTimeSeconds() + ' -is:retweet -is:reply'
            $entry = [ordered]@{ start = $cursor.ToString('o'); end = $sliceEnd.ToString('o') }
            try {
                $response = Invoke-RestMethod -Uri 'https://api.twitterapi.io/twitter/tweet/advanced_search' -Headers @{ 'X-API-Key' = $key } -Body @{ query = $query; queryType = 'Latest' } -Method Get -TimeoutSec 30
                $tweets = @($response.tweets)
                $accepted = 0
                foreach ($tweet in $tweets) {
                    $id = [string]$tweet.id
                    if ([string]::IsNullOrWhiteSpace($id)) { $id = $tweet | ConvertTo-Json -Compress -Depth 20 }
                    if ($seen.Add($id)) { $posts.Add($tweet); $accepted++ }
                }
                $entry.status_code = 200
                $entry.returned = $tweets.Count
                $entry.accepted = $accepted
                # Advanced-search cursors are documented as unreliable; only
                # flag a slice when it actually reaches the 20-post result cap.
                if ($tweets.Count -ge 20) {
                    $entry.warning = 'Result cap may have been reached; rerun this interval with a shorter slice.'
                }
            } catch {
                $entry.status_code = $null
                $entry.error = $_.Exception.Message
                $totalErrors++
            }
            $slices.Add([pscustomobject]$entry)
        }

        $folder = Join-Path $root $date
        New-Item -ItemType Directory -Force -Path $folder | Out-Null
        $document = [ordered]@{
            metadata = [ordered]@{
                source = 'twitterapi.io advanced_search'; timezone = 'Asia/Manila'
                window_start = $start.ToString('o'); window_end = $end.ToString('o')
                slice_minutes = 5; tweet_count = $posts.Count; slices = $slices
            }
            data = @($posts | Sort-Object createdAt)
        }
        $path = Join-Path $folder ('tweets_{0:D2}00_{1:D2}00.json' -f $hour[0], $hour[1])
        [System.IO.File]::WriteAllText($path, ($document | ConvertTo-Json -Depth 30), [System.Text.UTF8Encoding]::new($false))
        $totalPosts += $posts.Count
        Write-Output "$date $($hour[0]):00-$($hour[1]):00: $($posts.Count) posts"
    }
}

Write-Output "Collection complete: $totalPosts posts; $totalErrors failed slices."
