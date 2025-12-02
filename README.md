# GEPAdantic

<p align="center">
  <img src="https://github.com/pdoubleg/gepadantic/blob/main/assets/gepadantic.png" alt="GEPAdantic" width="500">
</p>

GEPA-driven prompt optimization for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents.

> [!NOTE]
> There is at least one other repo working on this same thing. See this [issue](https://github.com/pydantic/pydantic-ai/issues/3179) for more info. Unlike the project noted in the issue, which is a full re-write of GEPA, here we rely on the canonical GEPA api and simply provide a bridge between the two frameworks.

## About

This library combines pydantic's data validation with GEPA's prompt optimization algorithm. It does this by implementing a GEPA [adapter](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py) allowing a pydantic-ai agent to be plugged into GEPA's optimization api.

