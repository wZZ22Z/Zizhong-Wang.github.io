# wzz22z.github.io

Personal academic website of Zizhong Wang, built with the [Academic Pages](https://github.com/academicpages/academicpages.github.io) Jekyll template (MIT License) and deployed to GitHub Pages by `.github/workflows/jekyll.yml`.

- Site settings and sidebar: `_config.yml`
- Header menu: `_data/navigation.yml`
- Home page: `_pages/about.md`; CV menu opens `files/Zizhong_Wang_CV.pdf` directly; `_pages/cv.md` redirects legacy `/cv/` and `/resume` links to the same PDF.
- Publications: `_publications/`; blog posts: `_posts/YYYY-MM-DD-title.md`

## Blog

- 原始材料入口：旁边的 `../Blog/` 文件夹。每篇文章一个子文件夹，正文放 `post.md`、图片放 `img/`；复制 `../Blog/_template/` 即可开始。告诉我文章文件夹名后，由我整理到网站，具体说明见 `../Blog/README.md`。
- 博客入口：`/blog/`，页面模板为 `_pages/blog.html`；文章按年份归档，最新文章在前。
- 博客列表仅展示文章标题和 Keywords；关键词来自文章顶部的 `tags`，列表不展示阅读时长、发布日期或摘要。
- 写作模板：`_drafts/blog-post-template.md`，包含标题、摘要、标签，以及问题、方法、观察和参考资料等可选章节。
- 新建文章：复制模板到 `_posts/`，例如 `_posts/2026-11-12-my-first-post.md`，然后修改内容。文件名必须以 `YYYY-MM-DD-` 开头。
- 正文使用 Markdown，支持标题、列表、链接、图片、引用、表格和带语法高亮的代码块。
- `_drafts/` 中的文件不进入正常构建；可用 `bundle exec jekyll serve --drafts` 在本地预览草稿。
- 正式文章构建后会自动出现在 Blog 列表中，地址格式为 `/blog/YYYY/MM/DD/your-post-slug/`。线上发布沿用现有 GitHub Pages 流程。
