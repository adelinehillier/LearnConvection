using Documenter
using LearnConvection

makedocs(
    sitename = "LearnConvection",
    format = Documenter.HTML(),
    modules = [LearnConvection],
    pages = [
        "Home" => "index.md",
        "Library" => "main_function_index.md"
        # "Library" => Any[
        #     "Main" => "main_function_index.md",
        #     "GaussianProcess module" => "gaussian_process_function_index.md",
        #     "Data module" => "data_function_index.md"
        # ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/adelinehillier/LearnConvection.git"
)
