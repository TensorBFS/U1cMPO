using U1cMPO
using Documenter

DocMeta.setdocmeta!(U1cMPO, :DocTestSetup, :(using U1cMPO); recursive=true)

makedocs(;
    modules=[U1cMPO],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    repo="https://github.com/Wei Tang/U1cMPO.jl/blob/{commit}{path}#{line}",
    sitename="U1cMPO.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
