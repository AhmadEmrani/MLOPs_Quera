{{/*
Expand the full name of the release.
*/}}
{{- define "promiseops.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}
{{/*
Common labels (includes selector labels).
*/}}
{{- define "promiseops.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "promiseops.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
{{/*
Selector labels – used by Deployment, api-service, and metrics-service only.
Do NOT use for batch pods; they get their own inline labels in cronjob.yaml.
*/}}
{{- define "promiseops.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: promiseops
{{- end }}
{{/*
Validate image digest and return the full image reference.
Fails at render time if digest is empty, not sha256:<64 hex>, or the all-zeros placeholder.
*/}}
{{- define "promiseops.image" -}}
{{- $d := required "set image.digest to sha256:<64 lowercase hex>" .Values.image.digest -}}
{{- if not (regexMatch "^sha256:[0-9a-f]{64}$" $d) -}}
{{- fail "image.digest must match sha256:<64 lowercase hex chars>" -}}
{{- end -}}
{{- if eq $d "sha256:0000000000000000000000000000000000000000000000000000000000000000" -}}
{{- fail "image.digest is the all-zeros placeholder; CI must supply a real digest with --set" -}}
{{- end -}}
{{- printf "%s@%s" .Values.image.repository $d -}}
{{- end }}
