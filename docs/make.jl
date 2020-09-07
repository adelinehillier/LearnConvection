using Documenter
using LearnConvection

makedocs(
    sitename = "LearnConvection",
    format = Documenter.HTML(),
    modules = [LearnConvection]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/adelinehillier/LearnConvection.git"
)
