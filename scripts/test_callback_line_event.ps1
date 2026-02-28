# scripts/test_callback_line_event.ps1
# ASCII-only JSON using \uXXXX escapes to avoid any PowerShell/file encoding issues.

$uri = "http://127.0.0.1:5000/callback"

# "大浴場は何時まで入れますか？" in \u escapes:
# 5927 6D74 5834 306F 4F55 6642 307E 3067 5165 308C 307E 3059 304B FF1F
$json = @'
{
  "destination": "Uxxxxxxxx",
  "events": [
    {
      "type": "message",
      "message": {
        "type": "text",
        "id": "1",
        "text": "\u5927\u6D74\u5834\u306F\u4F55\u6642\u307E\u3067\u5165\u308C\u307E\u3059\u304B\uFF1F"
      },
      "timestamp": 0,
      "source": { "type": "user", "userId": "Utest" },
      "replyToken": "00000000000000000000000000000000",
      "mode": "active"
    }
  ]
}
'@

# Force UTF-8 bytes
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)

try {
  $res = Invoke-WebRequest -Method Post -UseBasicParsing $uri `
    -ContentType "application/json; charset=utf-8" `
    -Body $bytes `
    -ErrorAction Stop

  Write-Host ("STATUS={0}" -f $res.StatusCode)
  if ($res.Content) { Write-Host $res.Content }
} catch {
  if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
    Write-Host ("STATUS={0}" -f ([int]$_.Exception.Response.StatusCode))
  } else {
    Write-Host "STATUS=ERR"
  }
  Write-Host $_.Exception.Message
}