$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:EDGE_NODE_URL="http://127.0.0.1:8001/edge/api/generate"
$env:CLOUD_NODE_URL="http://127.0.0.1:8001/cloud/api/generate"
$env:PROMETHEUS_URL="http://127.0.0.1:8001/api/v1/query"

# Start Mock Infrastructure
Start-Process -NoNewWindow -FilePath "D:\ProgramFiles\anaconda3\envs\nt549-08-CapHuuTu\python.exe" -ArgumentList "mock_infrastructure.py"

Start-Sleep -Seconds 3

# Start Gateway App
Start-Process -NoNewWindow -FilePath "D:\ProgramFiles\anaconda3\envs\nt549-08-CapHuuTu\python.exe" -ArgumentList "-m uvicorn app:app --port 8000"

Start-Sleep -Seconds 5

Write-Host "Sending test request..."
$body = @{ query = "Bệnh nhân gút có hạt tôphi bị loét, kèm suy thận mãn tính giai đoạn 3 thì nên điều trị như thế nào?" } | ConvertTo-Json
Invoke-WebRequest -Uri "http://127.0.0.1:8000/ask" -Method Post -Body $body -ContentType "application/json" | Select-Object -ExpandProperty Content

Write-Host "`nTest complete. You can terminate the background processes using Task Manager or by killing python.exe."
