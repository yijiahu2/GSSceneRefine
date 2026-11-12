# GSSceneRefine Windows native loopback-only single-file model HTTP service.
# No administrator rights, Python runtime, Node.js, or data conversion required.
# GET /model.ply ONLY serves the exact model specified at startup. Never serves directories.
param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [int]$Port = 8766,
    [string]$AllowedOrigin = 'http://127.0.0.1:8765'
)
$ErrorActionPreference = 'Stop'
$full = [System.IO.Path]::GetFullPath($ModelPath)
if (-not [System.IO.File]::Exists($full) -or [System.IO.Path]::GetExtension($full).ToLowerInvariant() -ne '.ply') {
    throw 'The configured Gaussian .ply file does not exist.'
}
$listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
$listener.Start()
Write-Output "GSSceneRefine Windows model server ready on http://127.0.0.1:$Port/model.ply"
Write-Output "Allowlisted PLY: $full"
$UTF8 = New-Object System.Text.UTF8Encoding($false)
function Send-Headers($stream, [int]$status, [string]$reason, [long]$length, $extra) {
    $header = "HTTP/1.1 $status $reason`r`nContent-Length: $length`r`nConnection: close`r`nX-Content-Type-Options: nosniff`r`n"
    foreach ($key in $extra.Keys) { $header += "${key}: $($extra[$key])`r`n" }
    $header += "`r`n"
    $data = $UTF8.GetBytes($header)
    $stream.Write($data,0,$data.Length)
}
try {
  while ($true) {
    $client = $listener.AcceptTcpClient()
    try {
      $client.NoDelay = $true
      $client.ReceiveTimeout = 10000
      $stream = $client.GetStream()
      $reader = New-Object System.IO.StreamReader($stream,$UTF8,$false,1024,$true)
      $requestLine = $reader.ReadLine()
      if (-not $requestLine) { continue }
      $parts = $requestLine.Split(' ')
      if ($parts.Count -lt 2) { continue }
      $method = $parts[0].ToUpperInvariant()
      $target = $parts[1].Split('?')[0]
      $headers = @{}
      for ($n=0;$n -lt 100;$n++) {
        $line = $reader.ReadLine()
        if (-not $line) { break }
        $colon=$line.IndexOf(':')
        if ($colon -gt 0) { $headers[$line.Substring(0,$colon).Trim().ToLowerInvariant()] = $line.Substring($colon+1).Trim() }
      }
      $cors = @{'Cache-Control'='private, max-age=3600'; 'Vary'='Origin'}
      $origin = $headers['origin']
      $hostValue = $headers['host']
      if ($hostValue -notin @("127.0.0.1:$Port","localhost:$Port")) {
          Send-Headers $stream 403 'Forbidden' 0 $cors;continue
      }
      if ($origin -and $origin -ne $AllowedOrigin -and $origin -ne $AllowedOrigin.Replace('127.0.0.1','localhost')) {
          Send-Headers $stream 403 'Forbidden' 0 $cors;continue
      }
      if ($origin) { $cors['Access-Control-Allow-Origin']=$origin }
      if ($method -eq 'OPTIONS') {
        $cors['Access-Control-Allow-Methods']='GET, HEAD, OPTIONS'
        $cors['Access-Control-Allow-Headers']='Range, If-None-Match'
        Send-Headers $stream 204 'No Content' 0 $cors;continue
      }
      if ($method -notin @('GET','HEAD')) { Send-Headers $stream 405 'Method Not Allowed' 0 $cors;continue }
      if ($target -eq '/health') {
        $body = $UTF8.GetBytes('{"status":"ok","model_allowlisted":true}')
        $cors['Content-Type']='application/json'
        Send-Headers $stream 200 'OK' $body.Length $cors
        if ($method -eq 'GET') { $stream.Write($body,0,$body.Length) }
        continue
      }
      if ($target -ne '/model.ply') { Send-Headers $stream 404 'Not Found' 0 $cors;continue }
      $file=New-Object System.IO.FileStream($full,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::Read)
      try {
        [long]$size=$file.Length
        if ($size -lt 1) { Send-Headers $stream 500 'Empty File' 0 $cors;continue }
        $etag='"'+$size+'-'+[System.IO.File]::GetLastWriteTimeUtc($full).Ticks+'"'
        $cors['ETag']=$etag
        $cors['Accept-Ranges']='bytes'
        $cors['Content-Type']='application/octet-stream'
        if ($headers['if-none-match'] -eq $etag -and -not $headers['range']) {
          Send-Headers $stream 304 'Not Modified' 0 $cors;continue
        }
        [long]$start=0;[long]$end=$size-1;$partial=$false
        if ($headers['range']) {
          if ($headers['range'] -notmatch '^bytes=(\d+)-(\d*)$') {
            $cors['Content-Range']="bytes */$size";Send-Headers $stream 416 'Range Not Satisfiable' 0 $cors;continue
          }
          $start=[long]$Matches[1]
          if ($Matches[2]) { $end=[Math]::Min([long]$Matches[2],$size-1) }
          if ($start -gt $end -or $start -ge $size) {
            $cors['Content-Range']="bytes */$size";Send-Headers $stream 416 'Range Not Satisfiable' 0 $cors;continue
          }
          $partial=$true
          $cors['Content-Range']="bytes $start-$end/$size"
        }
        [long]$remaining=$end-$start+1
        Send-Headers $stream $(if($partial){206}else{200}) $(if($partial){'Partial Content'}else{'OK'}) $remaining $cors
        if ($method -eq 'GET') {
          [void]$file.Seek($start,[System.IO.SeekOrigin]::Begin)
          $buffer=New-Object byte[] (1024*1024)
          while ($remaining -gt 0) {
            $want=[int][Math]::Min([long]$buffer.Length,[long]$remaining)
            $read=$file.Read($buffer,0,$want)
            if ($read -le 0) { break }
            $stream.Write($buffer,0,$read)
            $remaining -= $read
          }
        }
      } finally { $file.Dispose() }
    } catch [System.IO.IOException] {
      Write-Warning "Client disconnect or IO error: $($_.Exception.Message)"
    } catch {
      Write-Warning "Request failed: $($_.Exception.Message)"
    } finally { $client.Close() }
  }
} finally { $listener.Stop() }
