package util

const (
	// Status of the detection task
	STATUS_PENDING int = iota
	STATUS_RUNNING
	STATUS_DONE
	STATUS_ERROR
	STATUS_NOT_FOUND
)

var Status2Str = map[int]string{
	STATUS_PENDING:   "Pending",
	STATUS_RUNNING:   "Running",
	STATUS_DONE:      "Done",
	STATUS_ERROR:     "Error",
	STATUS_NOT_FOUND: "Not Found",
}
