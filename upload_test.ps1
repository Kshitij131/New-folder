$uri = "http://localhost:8000/upload"
$clientsFile = "clients.csv"
$workersFile = "workers.csv"
$tasksFile = "tasks.csv"

$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$bodyLines = (
    "--$boundary",
    "Content-Disposition: form-data; name=`"clients`"; filename=`"clients.csv`"",
    "Content-Type: text/csv$LF",
    (Get-Content $clientsFile -Raw),
    "--$boundary",
    "Content-Disposition: form-data; name=`"workers`"; filename=`"workers.csv`"",
    "Content-Type: text/csv$LF",
    (Get-Content $workersFile -Raw),
    "--$boundary",
    "Content-Disposition: form-data; name=`"tasks`"; filename=`"tasks.csv`"",
    "Content-Type: text/csv$LF",
    (Get-Content $tasksFile -Raw),
    "--$boundary--$LF"
) -join $LF

Invoke-RestMethod -Uri $uri -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $bodyLines
