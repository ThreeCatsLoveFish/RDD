package main

import (
	"io"
	"os"

	"github.com/ThreeCatsLoveFish/RDD/api"
	"github.com/ThreeCatsLoveFish/RDD/template"

	"github.com/gin-gonic/gin"
)

func main() {
	// Initialize log file
	file, _ := os.Create("build/gin.log")
	gin.DefaultWriter = io.MultiWriter(file)

	// Initialize server
	router := gin.Default()
	api.LoadApi(router)
	template.LoadTemplate(router)
	router.Run(":8000")
}
